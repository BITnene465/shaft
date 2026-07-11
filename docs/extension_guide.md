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
6. 为真实 processor 输出实现并验证 batch 构造、rendered-token -> processed-token layout 与训练输入装配
7. 补模型、模板和单次 processor 契约测试

补充要求：

- 如果模型族有明确的 `language_model / vision_tower / aligner / generator` 分界，必须同步补充 `ModelModuleGroups`。
- 不要把模块名前缀写死在 `freeze.py` 或 `finetune.py` 中，冻结分组必须由模型族元信息声明。
- 多模态模型必须通过 `ProcessorPolicy` 统一声明四件事：`build_batch()` 的 processor 参数、
  `build_token_layout()` 的精确 token 映射、`assemble_training_inputs()` 的字段复制/重排，以及
  `cost_semantics_signature()` 的版本化成本语义。只有 processor
  输出与 chat template tokenizer 完全一致时才能使用 `identity`；存在图像 placeholder expansion、
  flattened vision rows、模型专属 sequence fields 或不同 media 参数时必须注册模型专用 policy。
- processor 的完整输出必须保留在 `ShaftProcessedBatch` 中。不要在 collator 增加
  `pixel_values/image_grid_thw/...` 白名单，也不要假设所有 tensor 的第 0 维都是 sample batch。
- policy 必须把 processor 的非 sequence 输出分别登记为 sample-aligned、whole-batch media 或 static；
  未声明字段会 fail fast。模型/processor 升级后新增输出时，应先确认其轴语义再更新 policy 和 DPO 测试。
- pixel-budget 支持只由 `ProcessorPolicy` 声明；通用 policy 默认不假设 processor 接受该参数。
- 声明 `supports_exact_image_cost=true` 的 policy 必须让 `cost_semantics_signature()` 覆盖 estimator 读取的
  所有 processor 参数及实现版本，例如 patch/merge、tile/crop、image token 和 pixel budgets。改变任一
  依赖状态必须改变 signature，并以参数化 conformance test 锁定；`supports_pixel_budget=false` 时收到
  pixel budget 必须 fail fast。禁止在 data cost provider 追加模型专属字段。
- token layout 必须做 exact validation。无法对齐时应在模型接入阶段失败，不允许改回逐 partial message
  重跑图片 processor，也不允许用长度差做近似 span。
- 如果新模型声称支持多图或视频，policy 与接入测试必须覆盖真实 media nesting、grid/patch 字段和 DPO
  pair 扩展；当前单图测试不能作为多图/视频支持证明。

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
- 训练模板还负责把消息角色编译为 canonical rendered-token supervision span；processor token expansion
  由模型 policy 映射，模板不处理图片。
- supervision plan 只能通过 `ShaftChatRenderer` 做一次完整 chat render 和纯文本 tokenization，不能接收
  processor/image/model adapter。HF 之外的 renderer 也应通过这两个 callable 适配，不扩宽 template API。
- 有稳定 role/message delimiter 的模板应复用 `ShaftDelimitedChatTemplate`；其它模板必须实现自己的
  full-render assistant-span compiler。基类在多轮 prefix supervision 下会 fail fast，不提供通用
  partial-message fallback。

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

## 9. 新增优化器 / scheduler / loss / loss_scale

### 9.1 必改位置

- `src/shaft/training/optimizer.py`
- `src/shaft/training/scheduler.py`
- `src/shaft/training/loss.py`
- `src/shaft/loss_scale/base.py`
- `src/shaft/loss_scale/mapping.py`

### 9.2 原则

- 统一走注册表
- 配置入口统一在 `TrainConfig`
- 不要在 pipeline 中硬编码新分支
- `loss_scale` 负责定义“哪些区段参与 loss”，不要把这类规则直接散写在 trainer 或模型 forward 中
- 当前 `loss_scale` 的落点是：
  - `template` 负责根据多轮消息角色生成 supervision plan，并直接产出单样本 `labels` / 可选 `loss_scale` tensor
  - `SFTCollator` 对每个 batch 只允许一次多模态 processor 调用，并消费模型 policy 生成的 token layout
  - `ShaftSFTTrainer` 负责把 `loss_scale` 从 batch 中剥离并传给 `loss.py`
  - `training/loss.py` 负责真正的加权 next-token loss 计算

## 10. 新增或修改冻结语义

### 10.1 必改位置

- `src/shaft/config/model.py`
- `src/shaft/config/normalize.py`
- `src/shaft/model/types.py`
- `src/shaft/model/freeze.py`
- `src/shaft/model/finetune.py`

### 10.2 原则

- 冻结规则和模型结构分组必须分开：
  - 规则层：`groups / prefixes / regex / trainable override`
  - 结构层：`language_model / vision_tower / aligner / generator`
- 训练执行与 adapter 导入校验必须共用同一份 `resolved finetune plan`，不要在 `loader / builder / export` 各自重复推导 target/modules_to_save
- `trainable_*` 优先级高于 `freeze_*`
- `full` 与 adapter 模式的冻结语义不同：
  - `full` 真正修改 `requires_grad`
  - `lora / dora / qlora` 只过滤自动展开的 adapter target，并补 `modules_to_save`
- 结构分组冻结必须按最具体前缀优先匹配，不能让宽前缀分组吞掉更具体的 `vision_tower / aligner / generator`
- 不要在 `pipeline`、`trainer`、`collator` 中写冻结逻辑
- 不要把某个模型族的模块名前缀硬编码进通用层
## 11. 新增导出能力

### 11.1 必改位置

- `src/shaft/export/hf.py`
- `src/shaft/cli/export.py`

### 11.2 原则

- 必须继续兼容 HF / PEFT 标准目录
- 不要引入自定义 metadata 目录
- 不要把发布逻辑塞进导出模块

## 12. 扩展时必须同步的文档

至少更新以下之一：

- `docs/architecture.md`
- `docs/module_reference.md`
- `docs/config_reference.md`
- `docs/extension_guide.md`
- `docs/online_eval_design.md`

如果新增的是用户会直接调用的能力，还要同步：

- `README.md`
- `docs/README.md`

## 13. 必跑测试

### 新模型族

- `tests/test_model_registry.py`
- `tests/test_model_processor_policy.py`
- `tests/test_template_registry.py`
- `tests/test_template_supervision.py`
- 必须覆盖真实或忠实 fake processor 的 image-token expansion、多轮 assistant span、无法精确对齐时失败
- 必须覆盖左右 padding、模型专属 processor 输出字段，以及存在时的 thinking/tool messages
- 必须断言 SFT/DPO 一个 batch 的多模态 processor 调用次数为 1，DPO chosen/rejected 复用同一份
  processor 输出且模型字段扩展正确
- 若声明多图/视频支持，必须增加对应真实 processor integration，不能只依赖单图 fake

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

## 14. 功能完成后的全局收口

- 一个 feature 基本完成后，不要直接提交。
- 先做一次“项目级别的收口 review”，重点看：
  - 是否出现重复状态源
  - 是否有逻辑落在错误层
  - 是否留下临时桥接代码或双轨实现
  - 是否需要在提交前先做一次小重构
- 这一步已经沉淀为项目 skill：
  - `.codex/skills/shaft-project/shaft-feature-review/SKILL.md`
- 这类 review 不是泛泛检查格式，而是明确回答：
  - 当前实现的真源在哪里
  - 是否还有冗余状态或冗余语义
  - 是否已经补齐必要测试和文档
