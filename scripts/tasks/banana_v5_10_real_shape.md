# v5.10 complete real shape reconstruction

独立真实数据源：`shape_context_reconstruction_real`，不是新增语义任务。
共享 `configs/prompts/pools/shape_context_reconstruction.v5.8.yaml` 的
`appearance / geometry / reconstruction` 三种 formulation。每个入选完整实例生成三份target，
共用一张context crop；不提供部分属性补救数据，不修改训练配置、catalog或采样权重。

## 输入与准入

- 显式传入用户确认的 `子属性文件名清单.txt`（939个文件名），不扫描名单外的子属性。
- 显式 compact raw 的 `json/` 和 `images/`，不自动使用旧 `shaft/data/raw` 副本。
- 复用已验收grounding的 `selection/{sources.json,tests.json,train.txt,metadata.json}`，
  按ID、SHA256、pHash<=6排除real_v1/real_v2/canonical测试候选。
- 当前43份名单文件在此前隔离目录，留在缺失报告中，不恢复；10份测试候选排除，886份参与筛选。
  若grounding指纹中原本存在的文件消失或改变，报错，不静默少选。
- 完整表示符合当前各类型的条件字段合同，不等于所有shape都需要相同字段。
  `other`仅类型、`oval`无角点是合法合同。`subbbox`不参与。

`real_shape_contract.py` 为转换规则真源，复用维护中的shape schema校验器：

1. 去掉不适用于该类型的占位字段；none/complex仅移除空的style/color占位。
   未知字段、不完整必填属性、活动属性缺失都拒绝，不补默认值。
2. 不推断缺失角点，不把一个圆弧分隔转换成两个split corners，不把五点尾部压成三点。
3. 轮廓反向时同步交换圆角start/end，所有坐标不变。起点取归一化包围框中最靠近左上的语义角点
   （归一化x+y最小，y/x决胜），只旋转已有顺序。保留card填充区域和分隔序列。
   三点tail必要时整体反向，保持中间tip和原几何。
4. 拒绝退化、重复和自交的采样轮廓；这是保守的离散几何门禁，不是视觉正确性认证。
5. 复用context/proposal与0–999转换；先无图片预检量化结果，全部通过才生成图片，
   避免失败样本残留无引用媒体。再次检查坐标、完整覆盖、退化和自交。

## 增强和数据边界

原图不动，保留每个成功实例的干净crop；按seed465和sample ID的SHA256稳定排序，额外选择
floor(20%)生成JPEG副本，quality60–90均匀随机，4:4:4。JPEG解码后存PNG，避免二次有损保存。
不加blur/noise/resize，副本坐标、proposal、target均与clean完全相同。具体配方唯一真源为
`configs/data/preparation/banana_v5_10_real_shape.json`。

train-only；所有val为空。SFT行prompt为空，由现有pool运行时渲染。训练接入时需将新dataset
显式映射到既有shape pool，并绑定三个formulation路径，再单独讨论真实/合成权重。
**本次发布不会自动进入任何正在运行或旧版训练。**

## 可复现命令

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 uv run --no-sync python \
  scripts/tasks/prepare_banana_v5_10_real_shape.py \
  --raw-root /path/to/raw_data \
  --filename-list /path/to/raw_data/子属性文件名清单.txt \
  --grounding-root data/grounding_layout \
  --work-root data/.build/banana-v5.10-real-shape-final \
  --workers 50
```

根目录可以换，输出引用均相对路径，数据来源ID固定而非目录名。
`--limit 100`生成独立canary且禁止发布；每次使用新的work-root。
`--stage prepare`冻结输入，`--stage build`检查锁后构建，不允许更改锁中的代码/配方/环境。
从其他机器重建需相同原始文件、TXT、grounding冻结清单、代码和图像codec环境；
不能只凭“有原图”重建相同标注或测试隔离结果。

全部验收后发布 `data/shape_context_reconstruction_real`，存在同名目录时拒绝覆盖。
每次报告包含选择/拒绝原因、源/test/code/prompt/environment指纹，以及structured/SFT/媒体
全量校验和。发布前后核对原始图片和JSON的哈希。不会修改现有合成shape数据。

## 验收状态

- 6项focused测试通过：完整字段投影、无损轮廓反向、缺失/自交拒绝、合法other/oval、
  源变更门禁、真实crop到三种formulation及确定性JPEG配对。
- 含JPEG的canary：100个clean +20个JPEG，120张共享图、360条SFT，全量验收通过。
- 小批量以50/1进程、不同工作路径重建，完整内容hash均为
  `db7dafadc25f4dd3fb4e65f6d2acc1bb2efa6ad09b7bfc5d274032092b6a4260`。
- 全量已验收发布：14,172个clean实例+2,834个JPEG副本，共17,006张图片，
  每种formulation17,006行，总51,018条SFT。
- 14,470个原始shape中，源属性/几何门禁拒绝298个；选中实例全部通过裁剪与量化，
  无额外视图拒绝。拒绝只作用于派生数据，不改raw。
- 源目标重算、完整覆盖、顺时针/退化/自交门禁、JPEG配对、三种SFT对齐、
  全量媒体完整解码/哈希及生成前后source hash均通过。
- 正式内容SHA256：`ede5fcc59582f7d02bfda5c5f3b935239e0a095832ec48a1125f38fd7ab6140e`。
  报告：`data/shape_context_reconstruction_real/reports/reproduction_result.json`。
