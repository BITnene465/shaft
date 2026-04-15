# Shaft TODO（延后项与未开工项）

本文档记录当前明确“暂不展开”的能力，避免后续重构把主线拉散。

## 1. 当前主线

- 当前主线需求仍是：**Qwen3VL 的 SFT / DPO / HF-first 训练与部署闭环**。
- 因此，优先投入：
  - 训练主链路稳定性
  - HF 兼容 checkpoint / export / merge 工具链
  - 数据中心、模型适配、模板与推理链路收口

## 2. 暂缓项

### 2.1 第二个真实模型族接入

- 当前暂不把 `GLM/Gemma` 等第二个真实模型族接入主干。
- 原因：
  - 现阶段业务主线仍是 `Qwen3VL`
  - 过早为多模型做重抽象，容易引入空转层级
- 结论：
  - 保留当前 `model/template/policy` 扩展入口
  - 待真实第二模型需求出现时，再用真实接入来验证抽象边界

### 2.2 Reward Model（RM）子系统

- 当前暂不展开 RM 训练、RM 数据格式、RM 评估与 RM 加载链路。
- 原因：
  - 现阶段 RLHF 主线仍以 `SFT + DPO` 为主
  - PPO 仍处于暂停/非生产状态
- 结论：
  - RM 作为 PPO 恢复开发前的前置工程，单独立项处理
  - 相关未完成项继续以 [docs/ppo_todo.md](ppo_todo.md) 为补充说明

## 3. 工具链范围

### 3.1 已纳入当前范围

- HF 导出目录校验
- PEFT adapter -> HF full export 合并
- `scripts/export.py` 工具入口

### 3.2 暂不展开

- 模型发布/上传工具链
- Hub 发布自动化
- 非 HF 生态导出格式

说明：
- 当前导出/合并只接受 **HF/PEFT 标准目录**。
- 不引入额外中间格式，不生成自定义 metadata 目录，不复制已有 full checkpoint。
