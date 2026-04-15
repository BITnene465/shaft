# Shaft 扩展指南

本文档描述如何在当前框架中新增模型族、模板、数据源、算法、推理后端和导出能力。目标是：扩展能力时不破坏现有边界，不在错误层落逻辑。

## 1. 总原则

- 先判断扩展属于哪一层，再改代码。
- 先补测试，再补实现。
- 优先复用注册表和适配层，不要平行复制一套流程。
- 当前扩展前提是 `HF-first`。

## 2. 新增模型族

### 2.1 必改位置

- `src/shaft/model/<family>.py`
- `src/shaft/template/<family>.py`
- `src/shaft/model/policies.py`（如需新增 policy）
- `src/shaft/model/registry.py`
- `src/shaft/template/registry.py`

### 2.2 推荐步骤

1. 定义 `ModelMeta`
2. 实现对应 `ModelLoader`
3. 需要时补 `ModelGroup`
4. 注册 `processor policy` / `peft policy`
5. 实现模板
6. 补模型与模板测试

### 2.3 不要做的事

- 不要在 `pipeline` 里分支判断模型族
- 不要在 `collator` 里重做模板解析
- 不要在 `infer` 里重复实现模型族前处理规则

## 3. 新增模板

### 3.1 必改位置

- `src/shaft/template/<family>.py`
- `src/shaft/template/registry.py`

### 3.2 关键点

- 模板只负责 `messages -> prompt`
- 模板只负责 `token_ids -> text`
- 模板元信息放在 `TemplateMeta`

禁止：

- 在模板里做图像裁剪
- 在模板里做任务后处理
- 在模板里决定 generation 策略

## 4. 新增数据源或样本格式

### 4.1 必改位置

- `src/shaft/data/sources.py`
- `src/shaft/data/meta.py`
- `src/shaft/data/registry.py`
- `src/shaft/data/dataset.py`
- `src/shaft/data/collator.py`
- `src/shaft/data/center.py`（仅在多源装配行为变化时）

### 4.2 关键点

- 新 source 必须注册
- 解析错误要聚合输出
- 样本元信息统一使用 `dataset_name`
- 数据源配置进入运行时前，先解析为 `ShaftDatasetMeta`
- catalog 扩展通过：
  - `data.catalog_path`
  - `data.catalog_names`

### 4.3 不要做的事

- 不要在 `pipeline` 中重写 mixing
- 不要在 `training` 中解析样本字段
- 不要把数据源路径逻辑塞进算法层

## 5. 新增训练算法

### 5.1 必改位置

- `src/shaft/algorithms/<algo>.py`
- `src/shaft/algorithms/registry.py`
- `src/shaft/config/runtime.py`
- `src/shaft/config/algorithm.py`
- `src/shaft/config/normalize.py`
- `src/shaft/pipeline/sft.py` 或 `src/shaft/pipeline/rlhf.py`

### 5.2 选择落点

- `sft` 类算法：进入 `ShaftSFTPipeline`
- `dpo/ppo` 类 RLHF 算法：进入 `ShaftRLHFPipeline`

### 5.3 原则

- 算法只构建 trainer，不读 JSONL
- 算法专属强类型配置优先放到 schema 中
- 不要在算法层处理模型族模板细节

## 6. 新增推理后端

### 6.1 必改位置

- `src/shaft/infer/engine.py`
- `src/shaft/infer/schema.py`
- `src/shaft/infer/loader.py`

### 6.2 原则

- 后端实现必须接受统一的 `ShaftInferRequest`
- 返回统一的 `ShaftInferResponse`
- stage 编排逻辑仍放在 `ShaftInferPipeline`
- 推理后端不得私有维护一套独立 codec；统一复用共享 codec 层

禁止：

- 后端实现直接依赖某个具体业务脚本
- 在 engine 中写 stage 级上下文规则

## 7. 新增 codec（共享层）

### 7.1 必改位置

- `src/shaft/codec/registry.py`
- `src/shaft/codec/base.py`
- `src/shaft/codec/<name>.py`

### 7.2 原则

- codec 只做文本到结构化结果的变换
- codec 应允许失败，并给出可排障的错误
- codec 应支持尽量修复和提取“合理部分”，尤其是 JSON 类输出
- codec 不负责训练数据格式
- codec 不负责指标计算
- codec 不负责业务编排

建议输出统一结构，而不是直接返回裸对象：

```python
ShaftCodecResult(
    raw_text=...,
    parsed=...,
    valid=True,
    partial=False,
    error_type=None,
    error=None,
)
```

说明：

- `infer` 使用 codec 做推理后处理
- 在线 eval 复用同一套 codec
- 不允许在 `infer` 和 `eval` 各维护一套 JSON 修复/解析逻辑

## 8. 新增在线 eval metric

### 8.1 设计前提

- 只考虑单阶段在线 eval
- 支持多数据集、多任务
- 每个数据集只绑定一个 task

### 8.2 必改位置

- `src/shaft/metrics/registry.py`
- `src/shaft/metrics/base.py`
- `src/shaft/metrics/builtin.py`
- `src/shaft/training/online_eval.py`
- `src/shaft/config/training.py`（`EvalConfig` / `EvalDatasetPolicyConfig`）

### 8.3 关键点

- 在线 eval 不直接耦合 `infer pipeline`
- 按 `dataset_name` 路由到 dataset eval policy
- target 侧统一走 `target_adapter`，不要把 GT 强行序列化回文本再解析
- 每个 dataset 必须声明：
  - `prediction_codec`
  - `target_adapter`
  - `metrics`
  - `primary_metric`
  - `normalizer`
  - `weight`
- 最终只输出一个 `eval_final_score` 作为 best model 选择依据

### 8.4 不要做的事

- 不要把多阶段业务编排塞进 trainer
- 不要让 metric 直接处理原始模型输出字符串
- 不要在进度条中实时刷 per-dataset task metrics
- 不要让所有指标直接参与 `final_score`，只允许使用 `primary_metric`
- 不要在启用在线 eval 时使用采样式评估；best-model 选择必须保持确定性

## 9. 新增优化器 / scheduler / loss

### 8.1 必改位置

- `src/shaft/training/optimizer.py`
- `src/shaft/training/scheduler.py`
- `src/shaft/training/loss.py`

### 8.2 原则

- 统一走注册表
- 配置入口统一在 `TrainConfig`
- 不要在 pipeline 中硬编码新分支

## 10. 新增导出能力

### 9.1 必改位置

- `src/shaft/export/hf.py`
- `src/shaft/cli/export.py`

### 9.2 原则

- 必须继续兼容 HF / PEFT 标准目录
- 不要引入自定义 metadata 目录
- 不要把发布逻辑塞进导出模块

## 11. 扩展时必须同步的文档

至少更新以下之一：

- `docs/architecture.md`
- `docs/module_reference.md`
- `docs/config_reference.md`
- `docs/extension_guide.md`
- `docs/online_eval_design.md`
- `docs/project_skill.md`

如果新增的是用户会直接调用的能力，还要同步：

- `README.md`
- `docs/README.md`

## 12. 必跑测试

### 新模型族

- `tests/test_model_registry.py`
- `tests/test_template_registry.py`

### 新数据源 / mixing / collator

- `tests/test_data_sources.py`
- `tests/test_data_center.py`
- `tests/test_collator.py`
- `tests/test_mixing.py`

### 新算法 / pipeline

- `tests/test_pipeline_sft.py`
- `tests/test_pipeline_rlhf.py`
- 对应算法专属测试

### 新推理后端 / codec

- `tests/test_infer_loader.py`
- `tests/test_infer_pipeline.py`
- `tests/test_infer_cli.py`

### 新在线 eval metric / codec 共享层

- `tests/test_codec.py`
- `tests/test_online_eval.py`
- `tests/test_pipeline_sft.py`

### 新导出能力

- `tests/test_export_tools.py`
- `tests/test_export_cli.py`
- 必要时加 checkpoint 兼容测试
