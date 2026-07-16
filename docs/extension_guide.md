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
- `src/shaft/model/descriptor.py`（只有现有 descriptor facts 不足时才扩展）
- `src/shaft/model/resolution.py`（通常复用，不为新模型另写平行 resolver）
- `src/shaft/template/<family>.py`
- `src/shaft/model/policies.py`（如需新增 policy）
- `src/shaft/model/inference.py` 与模型族专用 inference policy（若开放推理）
- `src/shaft/model/registry.py`
- `src/shaft/template/registry.py`

### 2.2 推荐步骤

1. 定义 `ModelMeta`
2. 实现对应 `ModelLoader`
3. 需要时补带稳定 `name` 与 `hf_model_types` 的 `ModelGroup`；dense/MoE 等 variant 必须由 HF config
   descriptor 选择，产品名只作 catalog hint
4. 注册 `processor policy` / `peft policy`
5. 若开放推理，显式注册 family-owned `ShaftInferencePolicy`；未注册时保持 fail closed
6. 实现模板
7. 为真实 processor 输出实现并验证 batch 构造、rendered-token -> processed-token layout 与训练输入装配
8. 若开放 varlen，实现 family-owned `SequenceExecutionPolicy`，同时验证 position reset、全/线性 attention
   state isolation、media kwargs 路由、backend/kernel/version 与 runtime shim；通用 trainer 不写模型分支
9. 补模型、模板、descriptor variant、processor/inference 契约和 packed-vs-standalone
   logits/loss/gradient 测试

`ResolvedModelPlan` 是 artifact/variant/adapter/sequence contract 的单一决议入口。新增模型族只能注册
descriptor matcher 与 policy，不能在 pipeline、builder 或 loader 再按路径名推导 dense/MoE。matcher 必须
只消费 upstream config facts；无法确认的 variant 应失败，而不是选择一个“最常见”默认值。
生产模型默认 `uses_hf_artifacts=true`，任何不存在但符合 `namespace/repo` 的 locator 都按 Hub artifact
解析 config；不得用 `models/outputs/checkpoints` 等 namespace 黑名单绕过。只有不代表真实 artifact 的测试
fixture 才能显式设为 false。
adapter 初始化也必须进入同一个 plan：外部 adapter 的 base/profile、PEFT signature 与 state shape 不得由
loader 或具体模型族另写一套宽松校验。模型族只负责解析 target policy；PEFT 的持久化 canonicalization 和
exact state-key 验证由通用 builder 统一处理。

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
- 新 processor policy 必须同时声明 `ProcessorInputPolicy`；训练输入和生成输入分别通过 `training`、
  `generation` mode 请求 padding，不允许在 pipeline/collator/infer 中重新写裸 `padding_side`。
- 推理 media/messages/chat-template kwargs/pixel-budget 语义必须由
  `ModelMeta -> ShaftModelAdapter.inference_policy` 提供。通用 HF/vLLM adapter 不允许按 `model_type`
  特判；只支持训练、尚未声明推理契约的模型应在任何图片编码或网络调用前 fail closed。
- 新 inference policy 必须分别声明并测试本地/远端后端能力。自定义 infer adapter 还必须声明
  deadline/cancellation capability；无法安全抢占的本地 generate 不得用后台线程伪造 timeout。
- GRPO/rollout 若需要预缩放，也必须实现 `ProcessorPolicy.prepare_rollout_image()`；pipeline 只注入 callable，
  `GRPODataset` 不得按模型名选择 resize 逻辑。
- 声明 `supports_exact_image_cost=true` 的 policy 必须让 `cost_semantics_signature()` 覆盖 estimator 读取的
  所有 processor 参数及实现版本，例如 patch/merge、tile/crop、image token 和 pixel budgets。改变任一
  依赖状态必须改变 signature，并以参数化 conformance test 锁定。`supports_pixel_budget=false` 表示通用
  policy 明确不向 upstream processor 转发预算；由于全局数据配置带有默认预算，这一兼容路径不会报错。
  如果某个模型族要求严格拒绝预算，必须在自己的 policy 中建立显式 strict contract 和测试，不能改变通用
  默认语义。禁止在 data cost provider 追加模型专属字段。
- token layout 必须做 exact validation。无法对齐时应在模型接入阶段失败，不允许改回逐 partial message
  重跑图片 processor，也不允许用长度差做近似 span。
- 如果新模型声称支持多图或视频，policy 与接入测试必须覆盖真实 media nesting、grid/patch 字段和 DPO
  pair 扩展；当前单图测试不能作为多图/视频支持证明。
- 如果 upstream 顶层会把 language kwargs 透传给 vision/audio 子模块，兼容修复必须放在版本化模型 runtime
  adapter 中，并对 API drift fail closed；禁止在 collator/trainer 删除模型专属字段。

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

- 算法只准备/构建 trainer，不读 JSONL。新增算法实现 `prepare_trainer()` 返回
  `ShaftTrainerSpec`；pure-local validation/config/辅助模型准备在该阶段完成，pipeline 只在全 rank
  readiness consensus 后、status envelope 外调用 `spec.build()`
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

## 13. 新增 hook / interceptor

`hook()` 与 `interceptor()` 属于训练轨迹边界，不只是日志注册表。新增插件时必须先判断它是否会改变模型输入、
loss、梯度、optimizer/scheduler、随机数状态、数据游标或 Trainer 控制流：

- 纯观测插件必须在 decorator 上显式写 `trajectory_neutral=True`。该参数只接受真正的 Python `bool`，字符串
  `"true"/"false"`、整数和 truthy 对象都会在注册时拒绝。
- `trajectory_neutral=True` 是开发者对轨迹不变性的承诺，不是绕过校验的开关。插件只能读取事件并写日志、
  metric 或外部 telemetry；不得修改传入对象、消费训练 RNG 或改变 callback control。
- neutral observer 在某个 rank 的 `before_step`、`after_step` 或 `on_save` 抛错时，不在训练热路径增加全 rank
  collective：本 rank 只记录一次 warning，并在本次 run 的后续所有事件中禁用该 observer。该降级同样适用于
  FSDP/DeepSpeed 的 backend-native `on_save` callback。未声明 neutral 的 hook 仍 fail fast，不会被静默吞掉。
- 未声明 neutral 或声明为 `False` 的插件，在 SFT/DPO/GRPO 开启 checkpoint 或 exact resume 时 fail
  closed。neutral observer 可以维护可重置的 telemetry counter/cache；这些状态不进入 checkpoint，也不得
  反向影响训练轨迹。当前框架尚未提供 trajectory-affecting plugin `state_dict/load_state_dict` 的版本化恢复协议。
- pipeline 必须把 manager 中的实际插件实例交给 `ShaftTrainingResumeContract`，不能根据配置名重新实例化一份
  仅用于 fingerprint 的对象。插件顺序、实现、closure/声明状态与 neutral marker 都属于 contract。
- SFT/RLHF pipeline 的 `before` interceptor 是独立的 rank-local readiness 阶段：所有 rank 成功后才会进入
  collective-owning pipeline body。插件不得在 `before` 中自行调用 distributed collective，也不得给零参数
  pipeline 注入 `args/kwargs`；rank-local 异常会由该阶段收敛到所有 rank。`after` 只在 target 成功返回后执行。

最低测试责任：decorator 类型反例、manager 到 resume contract 的真实装配、实现/顺序漂移，以及 checkpoint
模式下缺失 neutral marker/non-neutral 插件拒绝。pipeline-level interceptor 还必须用 CPU/Gloo 两 rank 注入单
rank `before` 失败，证明 peer 不进入后续 collective。neutral Trainer hook 则必须注入单 rank observer 异常，
覆盖 `before_step`、`after_step` 和 backend-native `on_save`，证明 peer 仍能进入后续 collective 且 observer
只告警一次。仅测试 registry 中存在某个名字不够。

## 14. 必跑测试

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

## 15. 功能完成后的全局收口

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
