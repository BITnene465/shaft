---
name: shaft-extension
description: 在 Shaft 框架中新增模型族、模板、算法能力的标准化流程。
---

# Skill：模型/模板/算法扩展

## 触发场景
- 新增模型族（例如 qwen3vl 变种、glm4v、gemma）
- 新增模板（prompt/chat）
- 新增算法（DPO/PPO）

## 步骤
1. 在 registry 层声明注册器，不在 core 流程硬编码。
2. 模型扩展优先接入 `ModelMeta -> ShaftModelAdapter` 这一条链路，不在 loader/collator/infer 中重复做模板或 policy 解析。
3. 若新增 processor/peft 策略，先落在 `src/shaft/model/policies.py` 注册，再由模型元信息引用；多模态
   processor policy 必须统一声明 batch 构造、canonical rendered-token 到 processed-token 的精确 layout，
   以及 SFT/DPO 模型字段的复制/重排。完整输出经 `ShaftProcessedBatch` 传递，collator 不维护字段白名单；
   非 sequence 字段必须在 policy 中声明为 sample-aligned、whole-batch media 或 static，未知字段 fail fast。
4. 在对应配置 schema 中新增最小必填项。
5. 在 `src/shaft/model|template|algorithms` 新增实现。模板监督只接收 `ShaftChatRenderer`，必须使用一次
   full-render span compiler；不得取得 processor/image，也不得增加 partial-message fallback。
6. CLI/脚本只做参数透传和命令路由。

## 验收
- 切换配置可运行，不改核心训练逻辑。
- 相关单测通过：model/template/algorithm registry 测试。
- 新模型的 supervision contract 必须覆盖 image-token expansion 与多轮 assistant span；SFT/DPO 每个
  batch 只能调用一次多模态 processor，DPO 必须正确复用/扩展全部模型专属字段。无法精确对齐或装配
  sequence-aligned 字段时应显式失败，禁止 partial-image fallback。声称支持多图/视频时必须补对应
  真实 processor integration。
