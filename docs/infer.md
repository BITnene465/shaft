# Shaft 推理子系统

本文档描述 `src/shaft/infer` 的职责、关键接口和扩展边界。

说明：

- codec 已抽成共享层，供 `infer` 与在线 eval 共用。
- 共享 codec 的唯一公开入口是 `shaft.codec`，不再通过 `shaft.infer` 重新导出。
- 因此本文档中的 codec 边界应理解为“共享 decode 能力”，而不是 `infer` 私有能力。

## 1. 目标

- 提供统一的单模型推理封装。
- 支持多阶段推理编排。
- 允许本地 HF 与 vLLM OpenAI 兼容后端并存。
- 让 codec 负责文本到结构化结果的收口。

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

## 3. 执行模型

### 3.1 单阶段

1. 根据 `InferEngineConfig` 构建 engine。
2. 根据 stage prompt 生成 `ShaftInferRequest`。
3. 执行一次模型调用。
4. 把文本输出交给 codec。
5. 返回结构化结果和 trace。

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

### 4.2 `backend_options`

- 用于透传后端级参数。
- 不能把它扩展成模型专属全能配置桶。

## 5. Engine 接口

### 5.1 `hf_local`

- 直接加载本地 HF 模型和 processor。
- 适合离线、小规模或调试环境。

### 5.2 `vllm_openai`

- 通过 OpenAI 兼容 HTTP 接口访问 vLLM。
- `min_pixels/max_pixels` 会透传到 `mm_processor_kwargs`。

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
