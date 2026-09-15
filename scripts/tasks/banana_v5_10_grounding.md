# v5.10 Grounding JPEG 数据准备

维护入口 `prepare_banana_v5_10_grounding.py`，配方真源
`configs/data/preparation/banana_v5_10_grounding.json`。只消费用户冻结的 raw，不清洗或覆盖源图/标注。
输入需包含新包清洗后标注和补回的 3,306 份 v5.9 paper 增量，共 23,376 份 JSON。

## 环境与命令

使用 Shaft 环境，额外离线依赖 `uv pip install ImageHash==4.3.2`；构建记录 Pillow、NumPy、
ImageHash、SciPy 版本与代码/prompt 哈希。跨机器复现使用相同源内容和环境，而不是相同服务器路径。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python \
  scripts/tasks/prepare_banana_v5_10_grounding.py \
  --raw-root /path/to/raw_data \
  --test-image-dir /path/to/real_v1/img \
  --test-image-dir /path/to/real_v2/img \
  --test-manifest /path/to/vlm.test.json --test-raw-root /path/to/canonical_raw \
  --work-root data/.build/banana-v5.10-grounding --output-root data --workers 50
```

`--limit 8` 用于独立 work root 的 canary，禁止发布限量产物。
`--phase select/build/jpeg/sft/verify/publish` 支持显式分阶段执行；默认顺序执行全部阶段。
不要向有旧产物的 work root 重跑全流程。代码、prompt 或 recipe 在选样后变化会拒绝后续阶段。
输入原图/标注在整个构建期间必须冻结不变。

## 选样与测试隔离

- 完整匹配 raw JSON 与同名图片，只消费 shape/icon/image/line bbox，不消费 subbbox。
- 使用 canonical 175 测试清单和显式 real_v1/real_v2 图库；排除相同 ID、相同文件 SHA256、
  pHash Hamming<=6 候选。近似匹配保守退出训练选择，不修改 raw，也不宣称每个候选都是确证重复。
- canonical 相对路径失效时，仅允许从显式测试图库找同名图片，要求内容无歧义且尺寸与清单一致；
  不能静默跳过缺失测试图。当前 50 张 ppt_000x 图从 real_v1 解析。
- 冻结 selection 中的 sources/tests 指纹、train/val split、recipe、代码和 prompt 哈希。
- train-only，val 为空。无四类目标的已标注源只保留一条 native empty，不增强。

## 视图配额

以正样本源图为基数：native 1.0、clean resize 0.9、padding 0.1、JPEG 1.0、
轻度 blur/noise 合计0.25、density crop 0.15、hard negative<=0.03。
配额受可行性约束，不以复制不合格视图强行凑数。

- 缩放目标 1,000,000–2,000,000 pixels，factor32，最多线性2倍放大；native 不改尺寸。
  小图不足下限或近原生重复时，基础 builder 可能不生成 resize，必须报告实际产量。
- padding 每轴5%–25%，不对称；复用基础 builder 的画布上限和 bbox 精确变换。
- blur/noise 限制 L1，blur radius=max(0.4,短边×0.0004)，noise sigma=2/255，两者不叠加。
- JPEG quality：40–59占25%、60–79占35%、80–95占40%；每源独立确定性采样一个质量值，
  全数据比例是概率目标，不是精确配额。采用4:2:0；JPEG编码/解码后保存PNG，避免重复压缩。
- JPEG 选择同源 clean resize 优先，其次 clean padding，最后 native；源无正目标不生成。
  精确记录 clean twin ID，保持尺寸/实例完全一致，不叠加模糊噪声。native fallback不保证1M下限。
- 裁图复用完整目标包含规则，部分相交则拒绝裁图；hard negative不碰任何完整/部分目标。
- Seed465，prompt沿用grounding_layout.v5.8的运行时详细/简略池；SFT行prompt保持空。

## 发布验收

全量 structured/SFT 对齐、ID/媒体唯一、bbox有效、目标重算、JPEG clean twin一致、
每源恰好一份native和每正源恰好一份JPEG、全媒体解码/尺寸/内容哈希。通过后才将工作目录
移动到 `data/grounding_layout`；既有目录保存为 `grounding_layout.before_v5_10`。
最终计数真源是 `reports/reproduction_result.json`，基础几何 builder 的摘要不包含后追加JPEG视图。
不自动修改训练数据权重或其他任务目录；整版 v5.10 尚需其他任务完成后才能打tag。

## 2026-09-09 发布结果

从23,376份raw标注选样，按三个测试清单的ID/内容规则排除116份，保留23,260个源，
其中23,142个正源、118个原生空目标源。real_v1的175张和real_v2的250张均纳入门禁，
另覆盖canonical175清单；原始数据不修改。

已发布 `data/grounding_layout`，structured/SFT/media各78,514条，val为空：

| 视图 | 数量 |
| --- | ---: |
| native full | 23,260 |
| clean resize | 19,905 |
| padded full | 2,314 |
| JPEG full | 23,142 |
| mild blur/noise | 5,786 |
| density crop | 3,471 |
| hard negative | 636 |

JPEG三档数量40–59为5,769、60–79为8,015、80–95为9,358。
Clean/degraded resize的实际像素范围1,003,520–1,999,872，全为factor32对齐。
与理想配额的差额来自测试内容排除、resize可行性/近原生去重以及hard-negative可行性；不额外复制凑数。
全量目标重算、配对、媒体完整解码与尺寸检查通过。旧派生目录备份为
`data/grounding_layout.before_v5_10`，canary保留在独立work root，不进入正式训练目录。
Content SHA256：`83c9b5e0b32ca85cf1b5ec125b874eb91962a568630cb43cf194ce63a9c04e06`。
## 2026-09-09 cleanup note

旧`grounding_layout.before_v5_10`及已完成的work/canary目录已清理；正式grounding目录、
selection/source/test指纹与复现报告保留。下文旧备份路径为历史记录。
