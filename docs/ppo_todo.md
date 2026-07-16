# PPO 暂停清单（未完成项）

本文档记录 Shaft 中 PPO 路径当前“可 smoke、不可生产”的边界与后续待办。  
状态：**暂停开发（暂不作为训练主线）**。

## 1. 当前结论

- 当前主线训练建议：`SFT + DPO`。
- PPO 仅保留为框架能力占位和最小 smoke 回归。
- 不建议打开以下调试开关用于正式训练：
  - `rlhf.ppo.allow_untrained_reward_model=true`
  - `rlhf.ppo.allow_text_only_multimodal_ppo=true`

## 2. 未完成项（阻断生产）

1. 缺少真实 Reward Model（RM）
- 目前 PPO 路径没有接入经过训练的 RM。
- 默认通过 fail-fast 防止误用随机奖励头。

2. 多模态 PPO rollout 未完成
- 现用 TRL experimental PPO 为 text-only query 路径。
- 对 Qwen3VL 等多模态模型，缺少完整的图像输入 rollout/打分链路。

3. PPO 全参微调未开放
- 当前只允许 `lora/dora/qlora`，`full` 模式被显式拒绝。

4. PPO 续训能力未完成
- 当前 pipeline 中 `resume_from_checkpoint` 对 PPO 仍是限制态。PPO 不接入 SFT/DPO/GRPO DDP 路径的
  `committed_manifest` exact-resume 协议，因此 `train.save_strategy` 必须为 `no`，避免生成外观类似但不可恢复的 checkpoint；
  `save_final_model` 的 `best` 导出和 root final state 仍可使用。

## 3. 已实现的防误用保护

- 未显式允许时拒绝随机奖励头：`allow_untrained_reward_model=false`（默认）。
- 多模态模型默认拒绝 text-only PPO：`allow_text_only_multimodal_ppo=false`（默认）。
- PPO query 是 rollout generation 输入；`PPOCollator` 在类级默认请求模型 policy 的 `generation` mode，
  异长 query 使用 generation padding（默认 left），不继承普通训练 loss 的 right padding。
- 显存保护默认启用：
  - `value_model_mode=shared_backbone`
  - `reward_model_mode=adapter_disabled_policy`

## 4. 恢复 PPO 开发的准入条件（建议）

1. 先补 RM 链路
- 新增 RM 训练流程（可先基于偏好对数据）。
- 新增 `reward_model_name_or_path` 与加载校验。

2. 明确 rollout 范围
- 方案 A：先做文本 PPO（明确“不含图像”）。
- 方案 B：实现多模态 rollout（含图像输入到 policy/value/reward）。

3. 完成稳定性与可观测
- 增加 PPO 训练关键指标看板与异常告警（KL、reward、entropy、ratio）。
- 增加最小可复现集成测试（非 smoke）。

4. 再开放生产开关
- 只有在 RM 与 rollout 链路稳定后，才考虑在生产配置中启用 PPO。
