---
name: shaft-grounding-arrow-data-aug
description: 将箭头整图标注整理为 grounding 训练数据，并按方法约束执行 sliding window、density crop、hard negative 与轻成像增强。
---

# Skill：grounding arrow 数据增强整理

## 触发场景
- 需要把整图箭头标注整理为 grounding 训练数据。
- 需要离线生成 crop / negative / augmentation 视图，但不希望把某次任务的固定输入输出路径写死到实现里。
- 需要复用已经验证过的方法约束，而不是重新讨论增强策略。

## 用户需要先明确的信息
- 输入标注目录或文件。
- 输出目录。
- train / val / test 的切分边界。
- 目标样本 schema。
- 是否保留原始全图视图、是否保留 keypoints 到 `extra`。

## 方法约束

### 任务边界
- 当前方法面向 `grounding`。
- 主 target 只保留 `label + bbox`。
- 需要保留更完整的结构信息时，放入 `extra`，不要污染当前 target。

### 训练集视图
- `full_image`
- `sliding_window_crop`
- `density_crop`
- `hard_negative_crop`
- 轻成像增强：
  - `jpeg_compression`
  - `light_blur`

### 验证集视图
- 默认只保留 `full_image`
- 不做 crop 扩增
- 不做 hard negative
- 不做 jpeg / blur

### sliding window
- 使用多尺度 tile。
- 只保留完整落在 crop 内部的实例。
- 滑窗样本应设置最小实例数阈值，避免生成过多稀疏正样本。
- 没有完整实例的滑窗可进入 hard negative 候选池。

### density crop
- 以实例中心或高密区域为候选。
- 只保留完整包含实例的 crop。
- 需要同时限制：
  - 最小实例数
  - 最大实例数
  - 每个尺度的最大 crop 数

### hard negative
- 仅从几何上可信的无实例视图中采样。
- 空样本最终占比应受控，不能无限扩张。
- 若 hard negative 过多，应重采样削减，而不是全部保留。

### 轻成像增强
- 只对训练视图做。
- 建议保持轻量：
  - JPEG 压缩
  - 轻模糊
- 不要一开始叠加过重的复合增强。

### 去重
- 对实例集合相同、crop 高度重叠的样本做去重。
- 目标是防止近重复视图大量污染训练分布。

## 执行步骤
1. 读取用户指定的标注输入。
2. 明确 train / val 的不同增强策略。
3. 先生成几何视图，再生成轻成像增强。
4. 对 crop 视图执行实例完整性过滤、最小实例数过滤和去重。
5. 对 hard negative 执行比例控制。
6. 按目标 schema 写出结构化样本与图像产物。
7. 生成最小统计报告，至少包含：
   - 各 split 样本数
   - 各视图类型样本数
   - 空样本占比
   - 去重前后数量

## 验收
- 训练集与验证集的增强边界清楚，不混淆。
- 主 target 不含无关字段。
- crop 样本只保留完整实例。
- hard negative 占比受控。
- 输出目录可全量重建，不依赖旧产物增量拼接。
