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

### 2.3 Infer 子系统重构

- 当前 `infer` 模块已经能支撑基础单阶段/多阶段推理，但整体完成度仍然不够，设计也不够稳定。
- 当前问题：
  - `engine / pipeline / codec / stage` 的边界还不够干净
  - 多阶段编排能力偏薄，更多是最小闭环，而不是成熟框架设计
  - 运行时参数、后端透传参数和业务编排之间的分层还可以继续收敛
  - 现阶段更适合支撑通用推理和轻量脚本调用，不适合继续承接复杂业务语义
- 结论：
  - 暂不把 `infer` 继续快速扩成重型子系统
  - 后续如需投入，应先做一次正式架构设计与边界重审，再进入实现阶段
  - 在重构完成前，复杂离线业务优先放在 `scripts/` 侧解决，不要持续污染 `src/shaft/infer`

### 2.4 离线 Eval 框架集成

- 当前暂不集成重型离线评测框架，例如 `EvalScope`、`OpenCompass`、`VLMEvalKit` 等。
- 原因：
  - 当前业务主线仍是 `Qwen3VL` 的训练闭环，不是通用 benchmark 平台建设
  - 这些框架集成成本高，且会引入额外的数据格式、服务接口与结果汇总约束
  - 现阶段任务评估更适合通过 `scripts/` 侧的定制脚本完成，而不是进入训练内核
- 结论：
  - 当前只保留训练内的 `eval_loss` 与已落地的轻量在线 task metric
  - 重型离线评测框架的接入，待真实 benchmark/统一评测需求出现后再单独立项
  - 在此之前，任务级离线评估统一放在 `scripts/eval/` 或其他脚本侧方案中解决

### 2.5 Arrow SFT 在线 Eval 与 Best Model 选择

- 当前箭头 SFT 配置仍然使用：
  - `eval.metric_for_best_model = eval_loss`
- 原因不是能力缺失，而是这份配置还没有把在线 task eval 配齐：
  - `eval.online_metrics_enabled` 还没开
  - `eval.datasets` 还没为：
    - `grounding_arrow`
    - `keypoint_arrow`
    配置完整 policy

- 下一个版本必须完成：
  1. 给箭头 SFT 配置补齐在线 eval 数据集策略
  2. 对 `grounding_arrow` 与 `keypoint_arrow` 分别配置：
     - `prediction_codec`
     - `target_adapter`
     - `metrics`
     - `primary_metric`
     - `normalizer`
  3. 打开：
     - `eval.online_metrics_enabled = true`
  4. 让配置在 normalize 阶段自动收口为：
     - `eval.metric_for_best_model = eval_final_score`
     - `eval.greater_is_better = true`

- 当前不在这一轮展开：
  - 新的复杂结构化任务指标实现
  - 重型离线 benchmark 接入

说明：
- 这件事的目标不是简单把 best metric 从 `eval_loss` 改成字符串 `eval_final_score`。
- 必须先把在线 eval 的数据策略和可用 metric 配齐，否则 best model 语义是空的。
- 第一版可以先基于当前已有能力推进：
  - `exact_match`
  - `parse_success`
  后续再补更贴近任务质量的结构化 metric。

- 需要继续收口的一项实现细节：
  - `training / online eval / infer / offline eval` 对 `padding_side` 的需求不同：
    - 训练态仍以 `right padding` 为主
    - 生成态应统一使用 `left padding`
  - 当前已经把这件事收敛到共享 `build_processor_inputs(..., padding_side=...)` 真源中，但还没有上升为正式的 processor-input policy 语义。
  - 后续需要把“训练态 / 生成态 padding policy”进一步显式化，避免未来在某条新路径上重新出现局部补丁或遗漏 left padding 的问题。

- 需要继续收口的另一项 eval 语义：
  - 当前 `online eval` 的 pixel budget 仍然默认复用 `data.min_pixels / max_pixels`。
  - 训练输入预算和评估输入预算不应长期耦合：
    - 训练态 budget 更偏吞吐与稳定性
    - eval / online eval budget 更偏任务质量与可比性
  - 后续应把 pixel budget 正式分层为：
    - `data.*` 负责训练数据默认预算
    - `eval.*` 负责评估默认预算
    - `eval.datasets.<name>.*` 允许对特定任务做 per-dataset override
  - `eval_final_loss` 与 `eval_final_score` 应共享同一套 eval pixel budget 解析规则，避免 loss 和 generation score 在不同分辨率语义上比较。

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

## 4. 数据 Mixing 后续增强

- 当前已支持：
  - `static`
  - `epoch_refresh`
- 暂不在这一轮展开：
  - staged / schedule-based mixing
  - feedback-driven adaptive mixing
  - mixing policy / state / sampler 进一步拆层
  - 基于任务反馈的权重更新稳定化（EMA、最小步长、冷启动阶段）
  - batch 级长度分桶与 prompt/image 复杂度联合采样

说明：
- 当前先把 `static / epoch_refresh` 的 sampler 主路径做稳。
- 后续如果要继续投入，应优先把 mixing state / policy / sampler 明确拆层，而不是在 `data center` 里继续堆条件分支。

### 4.1 GRPO 与 mixing 刷新语义对齐

- 当前 `GRPO` 明确只支持：
  - `data.mix_refresh=static`
- 原因：
  - 现有实现依赖 TRL `GRPOTrainer` 自己的 prompt-repeat / grouped generation sampler
  - Shaft 自己的 `epoch_refresh` train sampler 当前不会传入 `GRPOTrainer`
  - 因此 `GRPO` 与 `epoch_refresh` 现在存在采样控制权冲突

- 暂不在这一轮展开：
  - 让 `GRPO` 直接复用现有 `ShaftMixedIndexSampler`
  - 通过临时 callback 或 dataset 重建去桥接 `epoch_refresh`

- 后续如果要支持 `GRPO + epoch_refresh`，正确方向是：
  - 单独设计 `GRPO-aware sampler`
  - 同时满足：
    - dataset mixing
    - prompt repeat
    - grouped generations
    - distributed sharding
  - 而不是继续在 pipeline 或 trainer 外层叠桥接逻辑

说明：
- 这件事的核心不是“放开配置”，而是重构采样主控权。
- 在没有 `GRPO-aware sampler` 之前，继续保持 `static-only` 是正确约束。

## 5. Data Center / Dataset 增强边界重构

- 当前暂不继续扩 `data center` 和 `dataset` 上的数据增强职责。
- 当前结论：
  - 训练期 `online transform` 仍可由 `data center -> dataset` 这条链统一编排。
  - 真正的重型离线增强，例如：
    - sliding window crop
    - density crop
    - hard negative 生成
    - 图像写盘与新 JSONL 产物生成
    继续放在 `scripts/tasks/prepare_*` 这类离线数据生产流程中处理。
- 暂不在这一轮展开：
  - `data center` 的 train/eval online transform 进一步拆分
  - 运行时轻量 offline transform 与数据生产型 offline augmentation 的正式架构拆分
  - `dataset` 层更细粒度的增强策略路由

说明：
- 当前不希望把数据资产生产逻辑继续塞进训练运行时主链。
- 后续如果要继续做这块，应先明确：
  - 哪些增强属于数据准备
  - 哪些增强属于训练运行时
  - 然后再决定是否重构 `data center / dataset / transforms` 的边界。
