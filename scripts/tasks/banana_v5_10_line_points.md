# Banana v5.10 line points

同一个 line reconstruction 任务的 `points` formulation，沿用
`configs/prompts/pools/line_context_reconstruction.v5.8.yaml`；物理 cohort 为
`line_context_points`。不新增 prompt，不要求真实数据具有 full 属性。

## 输入和配方

- 当前用户指定 compact raw 的 `json/` 与 `images/`。不是自动改用 `shaft/data/raw` 的旧副本。
- 已发布 v5.10 grounding 的 `selection/{sources.json,tests.json,train.txt,metadata.json}`。
  复用 ID、图像 SHA256、pHash<=6 的 real_v1/real_v2/canonical 排除结果；源 JSON/图片哈希
  必须与排除时一致。跨机器需要传递这些清单，或先用相同输入重建 grounding。
- 修正尺寸后的 V10 原始 `gt_standard/` 与 `img/`；已发布 v5.10 synthetic line 的
  `selection/train.jsonl` 和 `reports/input_checksums.json` 只提供身份、分层和校验和，
  路径真值仍每次从原始 GT 读取。该 selection 已排除 V10 val 及显式测试 ID。
- 版本配方 `configs/data/preparation/banana_v5_10_line_points.json`，seed465。
  synthetic 像素策略读取并冻结 `banana_v5_10_line.json`，不维护第二份增强参数真源。

保留所有符合几何合同的真实打点；空 points 不推断。合成补充选 15,000 条多路径候选，
按完整 line stratum 的容量平方根分配，SHA256 排序无放回抽取，增强稀有分层占比；不加入单路径。
数量是候选上限，若新 context 下量化碰撞会记录拒绝，不通过悄悄删点填满名额。

真实图保留干净 context crop，在成功生成的真实行中稳定抽取 floor(20%)，每行追加一份
JPEG quality 60–90 均匀随机副本，4:4:4。JPEG 解码后存 PNG，避免读取时再压缩，
不叠加模糊、噪声、缩放；副本尺寸、proposal、路径和 target 与干净行完全相同。
合成 crop 使用原定单操作 80% JPEG40–90 / 20%轻噪声，原始目标短边<16px保持clean；
合成 locator 四边各扩2px，raw bbox和路径保持不变。

所有目标只含 `type=line, parameters={is_single,points}`。不改源顺序，不删曲线采样点，
不按 bbox 去重；点量化碰撞只拒绝派生样本并记录。`is_single` 由路径数确定，不推断外观属性。

## 执行

先做 `--limit 20` canary（每个cohort最多20候选），使用独立工作目录。canary禁止发布。
全量命令，根目录可在其他机器替换：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 uv run --no-sync python \
  scripts/tasks/prepare_banana_v5_10_line_points.py \
  --raw-root /path/to/raw_data \
  --synthetic-root /path/to/regulated_layout_dataset_v10_20260830 \
  --grounding-root data/grounding_layout \
  --synthetic-cohort data/line_context_reconstruction \
  --work-root data/.build/banana-v5.10-line-points-final \
  --workers 50 --replace
```

`--stage prepare` 只冻结选择和输入指纹；`--stage build` 从冻结清单构建，校验代码、配方、
环境和输入字节。中途修改代码不能直接继续旧锁；必须另建工作目录重新准备。
并行度不参与采样随机性；图像codec/Pillow等版本记录于 reproduction.lock.json。
输出使用相对路径，不绑定服务器根目录。

只有全部验收通过才发布至 `data/line_context_points`；已有目录备份到
`data/line_context_points.before_v5_10`，不覆盖已存在的备份。失败保留工作目录用于排查。
不更新训练任务权重，不代表整个 v5.10 已发布，不自动创建 Git tag。

## 验收与状态

- 全量从 raw/GT 重算 ordered points，检查完整裁剪覆盖、尺寸及 target。
- structured 与 points formulation SFT 一一对应，全部媒体完整解码和哈希。
- 唯一 sample ID/media，JPEG 与 clean twin 几何/target对齐，无悬空或额外PNG。
- 空 val 表示 train-only。源拒绝记录在 lock，视图拒绝记录在 reports/rejected_views.json。
- 固定输入清单、代码、prompt、环境、选择清单及全内容指纹。

2026-09-09：小批量44行验收通过（20真实 + 20合成 + 4真实JPEG）。
相同冻结输入分别以50进程与1进程、不同工作路径重建44行，完整内容SHA256均为
`650a340b08dbf7a8df4048bc444945e45ed8a25fec4f5e2c05503d94f0fea999`。
新入口与共享V10准备逻辑的11项focused测试通过。
全量已发布并验收：187,727真实干净 + 37,545真实JPEG + 15,000合成，共240,272行。
原始候选187,921真实 + 15,000合成；另18条源几何不合格、194条真实量化碰撞仅记录拒绝，
不改raw、不删点。全量原始路径重算、完整裁剪覆盖、JPEG配对、structured/SFT对齐、
唯一媒体检查、全部图片完整解码及哈希均通过。
完整内容SHA256：`3f6748e6e2fa92a73b32dadeb2c19fe51c90a7ed4261ef939ec02a8be13277db`。
正式报告：`data/line_context_points/reports/reproduction_result.json`。
原派生目录备份：`data/line_context_points.before_v5_10`。
## 2026-09-09 cleanup note

旧`line_context_points.before_v5_10`及已完成的work/canary目录已清理；正式数据与所有生产复现锁
保留。下文旧备份路径为历史记录，重建请从完整原始输入重新prepare。
