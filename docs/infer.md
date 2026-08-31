# Shaft 推理子系统

本文档描述 `src/shaft/infer` 的职责、关键接口和扩展边界。

说明：

- codec 已抽成共享层，供 `infer` 与在线 eval 共用。
- 共享 codec 的唯一公开入口是 `shaft.codec`，不再通过 `shaft.infer` 重新导出。
- 因此本文档中的 codec 边界应理解为“共享 decode 能力”，而不是 `infer` 私有能力。

## 1. 目标

- 提供统一的单模型、单图或有序多图推理封装。
- 支持多阶段推理编排。
- 允许本地 HF 与 vLLM OpenAI 兼容后端并存。
- 让 codec 负责文本到结构化结果的收口。

当前公共 API 的执行单位是一个包含至少一张图片的样本；一次 `run()` 同步、顺序执行各 stage。框架不提供
原生多样本 batch、streaming、async queue 或 Shaft 自有在线服务层。外部 vLLM 服务负责服务端 batching，
但这不改变 Shaft 请求侧的单样本合同。

当前 infer stage 直接配置 `user_prompt_template`，尚未提供从训练 PromptSource pool 按
`formulation_id/variant_id` 显式选择 prompt 的公共接口。需要按属性集合推理时，调用方目前必须使用与对应
formulation 等义的 materialized prompt；不得声称 infer 已经自动理解 `A`、`B`、`A+B` 的组合关系。

## 2. 核心对象

### 2.1 配置对象

- `InferEngineConfig`
- `InferStageConfig`
- `InferPipelineConfig`

### 2.2 运行时对象

- `ShaftInferRequest`
- `ShaftInferResponse`
- `ShaftInferEngine`
- `ShaftInferPipeline`
- `ShaftInferStageResult`

### 2.3 辅助对象

- `HFLocalInferAdapter`
- `VLLMOpenAIInferAdapter`
- `ShaftInferExecutionControl`
- `ShaftInferAdapterCapabilities`

模型族侧由 `ModelMeta -> ShaftModelAdapter.inference_policy` 声明推理能力。policy 负责把统一 request
准备成后端输入，包括 media、image-first messages、chat-template 参数和 pixel budget 语义；通用
infer adapter 只负责执行。没有显式 inference policy 的模型默认 fail closed，不会套用 Qwen 行为。

`ShaftInferRequest.image_paths` 是 media 的规范真源，并按给定顺序进入 chat placeholder、HF processor 或
OpenAI `image_url`。旧 `image_path` 只保留为单图兼容入口，两者不能同时提供；多图请求访问单数属性会
明确失败，避免运行时出现两套 media 真源。显式 `messages` 必须用相同数量、相同顺序的 `type: image`
占位符。pipeline 对应使用 `run(image_paths=[...])`，CLI 使用可重复参数：

```bash
python scripts/infer.py --config configs/infer/pipeline_smoke.yaml \
  --image first.png --image second.png
```

### 2.4 大模型本地加载

`InferEngineConfig.device_map` 会透传到 HF `from_pretrained`。当设置为 `auto` 或显式
device map 时，`HFLocalInferAdapter` 不再把模型整体 `.to(device)` 到单卡，而是按 HF
已经生成的 `hf_device_map` 保留分片加载结果，并把输入张量移动到首个模型设备。

## 3. 执行模型

### 3.1 单阶段

1. 根据 `InferEngineConfig` 构建 engine。
2. 根据 stage prompt 生成 `ShaftInferRequest`。
3. 模型 inference policy 准备 media/messages/chat-template 参数。
4. adapter 在 deadline/cancellation 契约内执行一次模型调用。
5. 把文本输出交给 codec。
6. 返回结构化结果和 trace。

### 3.2 多阶段

1. 按 stage 顺序执行。
2. 每个 stage 都可以从已有上下文中读取字段。
3. 结构化结果会写入上下文，供后续 stage 使用。
4. `__trace__` 记录每个 stage 的耗时、重试、失败原因。

## 4. Stage 接口

`InferStageConfig` 关心：

- `name`
- `engine`
- `user_prompt_template`
- `arguments`
- `output_key`
- `system_prompt`
- `generation`
- `codec`
- `min_pixels`
- `max_pixels`
- `backend_options`
- `max_retries`
- `retry_backoff_seconds`
- `fail_fast`
- `timeout_seconds`

### 4.1 `output_key`

- 若设置，stage 解码后的结果会写入共享上下文。
- 后续 stage 可以通过 prompt 模板读取该字段。

### 4.2 Prompt 参数

- stage 的 `arguments` 使用与训练 prompt pool 相同的 schema；模板语法只有 `{{ name }}` 与
  `{{ name | json }}`。
- `{{ name }}` 只适用于 `string/enum`；结构化结果应声明为 `json` 并使用 `json` filter。
- 普通 JSON 单花括号保持原样。旧 Python format 写法 `{stage_out}` 已移除，配置加载会给出迁移错误。
- renderer 在调用 engine 前完成严格的缺参、额外参数、类型和有限数校验。之后由模型 inference policy
  按该模型的 media/message 契约构造后端输入；pipeline 不感知模型族。
- stage trace 的 `prompt_audit` 记录 renderer/version 与模板、参数、最终 prompt hash。

### 4.3 `backend_options`

- 用于透传后端级参数。
- 不能把它扩展成模型专属全能配置桶。
- 模型 inference policy 必须先校验这些参数；与 policy 自己负责的 media/pixel-budget 字段冲突时
  fail closed，不能静默形成两套真源。

## 5. Engine 接口

### 5.1 `hf_local`

- 直接加载本地 HF 模型和 processor。
- 适合离线、小规模或调试环境。
- processor 使用模型 `ProcessorInputPolicy` 的 `generation` mode，不在 infer 中写死 padding side。
- 本地 `model.generate()` 当前没有安全的抢占/取消边界。因此带 deadline 或 cancellation contract 的请求
  会在打开图片、processor 和 generate 之前抛出
  `ShaftInferExecutionControlUnsupportedError`；不会用后台线程制造“调用已超时但推理仍占资源”的泄漏。

### 5.2 `vllm_openai`

- 通过 OpenAI 兼容 HTTP 接口访问 vLLM。
- Qwen inference policy 不把 `min_pixels/max_pixels` 透传给 vLLM，而是在发送请求前按 Qwen
  像素预算重采样图片，避免不同 vLLM 版本的 multimodal processor 参数口径漂移。其他模型不会继承
  这个约定。
- Qwen35/36 inference policy 默认在请求中加入
  `chat_template_kwargs={"enable_thinking": false, "preserve_thinking": false}`。如果确实要评估
  thinking 模式，应显式设置 `template=qwen35vl_thinking` 或通过 stage/request 的
  `backend_options.chat_template_kwargs` 覆盖。
- `configs/infer/qwen36_vllm_example.yaml` 提供了最小 Qwen3.6 vLLM OpenAI 示例。vLLM 服务本身
  仍需以支持 `qwen3_5` 架构的 Transformers / vLLM 环境启动。
- Shaft 当前不支持 Qwen3.5/3.6 MTP。通过 Shaft SFT/merge 生成的权重不保证包含 `mtp.*` draft head，
  即使 `config.json` 仍有 `mtp_num_hidden_layers` 也不能启用 vLLM/SGLang MTP speculative decoding。
  外部部署入口中的 speculative 参数不进入 Shaft 支持合同。官方原始权重在外部服务中启用 MTP 属于后端
  自有能力，不得把结果外推到 Shaft 训练产物。
- 当前标准推理环境以 `uv.lock` 为准，已经验证到 `vllm==0.19.1` 与
  `transformers==5.10.1`。部署系统必须用同一份 lock 构建等价环境，并自行维护镜像、服务编排和发布生命周期；
  本仓库不提供生产部署 Dockerfile、vLLM launcher 或业务 smoke wrapper。
- 只启动 vLLM 仍不足以复现实验效果；业务调用侧还必须对齐 prompt pool、Qwen pixel budget
  smart resize、generation 参数和共享 codec/JSON 解析策略。

### 5.3 Deadline 与 cancellation

- `InferStageConfig.timeout_seconds` 会在 stage 开始时转换成一个 absolute monotonic deadline；同一个
  deadline 贯穿该 stage 的所有 attempt、retry backoff、HTTP connect 和 response-body read，不会在每次
  retry 时重新获得完整预算。
- pipeline 在调用前检查 engine capability，把 control 注入 `ShaftInferRequest.execution`，并在
  `engine.run()` 返回后、共享 codec decode 完成后分别再次 checkpoint。后置 checkpoint 同时约束 backend
  与解析阶段，不是旧式“调用已经阻塞完成后才比较耗时”。
- vLLM adapter 的实际 HTTP timeout 取
  `min(engine.request_timeout_seconds, stage_remaining_seconds)`；读取 response body 时按剩余 absolute
  deadline 分块并更新 socket timeout，超时后关闭 response。
- `ShaftInferPipeline.run(..., cancellation_event=event)` 公开 cooperative cancellation。retry backoff
  使用 `Event.wait()`，因此 cancellation 不必等待完整 backoff。adapter 若没有声明 cancellation 能力，
  pipeline/engine 会在工作开始前 fail closed。cancellation 是 pipeline-global 终止信号，不受 stage
  `fail_fast` 控制，也不会被记录成普通 stage failure 后继续后续 stage。
- 不允许用 executor/background thread 包装无法抢占的本地 generate 来伪造 timeout；返回超时的同时仍有
  推理在后台运行属于资源泄漏。

### 5.4 部署边界与契约验收

本仓库只维护 `src/shaft/infer` 的推理合同和 task-local 评测工具，不维护生产镜像、vLLM 服务 launcher 或业务
部署 smoke。部署系统消费 Shaft 权重时，必须在自身受追踪的配置中固定 checkpoint/merge provenance、依赖
版本、prompt hash、image-first 消息顺序、Qwen pixel budget、thinking 开关、generation 参数和 parser 版本。
部署环境与本地评测出现差异时，应使用同一图片和同一完整请求体对两个 endpoint 做 raw-output A/B；不能用
另写的简化 smoke 代替正式调用合同。

## 6. Codec 设计（共享层）

codec 的职责是：

- 接收模型文本输出。
- 解析为结构化 payload。
- 在需要时做容错修复。

当前原则：

- codec 是稳定扩展点。
- stage 编排不是任务 DSL。
- codec 不负责训练数据规范。
- codec 不负责指标计算。

## 7. 文档化边界

### 允许

- 增加新 codec
- 增加新 infer backend
- 扩展 trace 字段
- 扩展 stage 级 generation 配置

### 禁止

- 在 `infer` 中写死某个具体业务任务
- 在 `codec` 中混入训练时模板逻辑
- 在 `engine` 中加入复杂任务后处理

## 8. 当前建议

- 通用推理能力尽量走 `src/shaft/infer`。
- 一次性离线业务脚本放在 `scripts/tmp/`，不要为了单次任务污染推理内核。
- 若一个业务需求需要大量编排语义，先判断是否真的该进入框架主干。
