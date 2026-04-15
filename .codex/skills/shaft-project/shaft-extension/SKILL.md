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
3. 若新增 processor/peft 策略，先落在 `src/shaft/model/policies.py` 注册，再由模型元信息引用。
4. 在对应配置 schema 中新增最小必填项。
5. 在 `src/shaft/model|template|algorithms` 新增实现。
6. CLI/脚本只做参数透传和命令路由。

## 验收
- 切换配置可运行，不改核心训练逻辑。
- 相关单测通过：model/template/algorithm registry 测试。
