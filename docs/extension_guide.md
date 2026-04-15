# Shaft 扩展指南（SFT / RLHF / 模型适配 / 数据系统）

本文档是后续 agent 的“改代码说明书”。目标是：新增能力时，改动最少、边界清晰、可测试。

## 0. 扩展前提：默认沿用 Hugging Face 生态

- Shaft 当前扩展默认前提是 **HF-first**。
- 新增模型族、算法、数据能力时，优先接入以下已有生态：
  - `transformers`
  - `peft`
  - `trl`
  - HF 兼容保存目录与权重加载方式
- 如果某项需求只能通过引入非 HF 生态才能完成，必须先在架构层重新确认边界；不要直接把新生态的概念和兼容逻辑塞进现有主干。
- 换句话说：当前扩展目标是“在 HF 生态内做强做稳”，而不是把 Shaft 演化成多生态兼容层。

## 1. 新增训练算法（例如新 RL 算法）

### 1.1 必改文件
1. `src/shaft/algorithms/<algo>.py`
2. `src/shaft/config/schema.py`（新增结构化配置）
3. `src/shaft/config/normalize.py`（新增校验）
4. `src/shaft/pipeline/rlhf.py` 或 `src/shaft/pipeline/train.py`（按算法归属接线）
5. `src/shaft/cli/`（如需新命令或新参数）
6. `tests/`（单测 + pipeline smoke）

### 1.2 设计原则
- 如果算法属于 RLHF（依赖 preference/reward），放到 `shaft_rlhf` 流水线，不要塞进 `shaft_train`。
- 算法实现只负责 trainer 行为，不负责读取 JSONL。
- 算法参数必须有强类型配置，不允许长期使用 `dict[str, Any]` 裸传。

### 1.3 DPO/PPO 已有范式
- DPO：`ShaftDPOTrainer`（TRL DPOTrainer），输入 pairwise（chosen/rejected）。
- PPO：`ShaftPPOTrainer`（TRL PPOTrainer），输入 query/prompt（在线 rollout）。
- 参考入口：
  - `src/shaft/algorithms/dpo.py`
  - `src/shaft/algorithms/ppo.py`
  - `src/shaft/training/rlhf.py`

### 1.4 当前能力边界（必须先读）
- DPO/PPO 均使用 TRL 内核，Shaft 主要负责配置映射、数据形态和流水线编排。
- PPO 默认数据格式为 query-only；当前实现有两条安全保护：
  - 未显式开启 `allow_untrained_reward_model` 时，拒绝使用随机奖励头启动训练。
  - 多模态模型默认拒绝进入 text-only PPO 路径，需显式开启 `allow_text_only_multimodal_ppo`（仅建议 smoke/debug）。
- PPO 当前仅支持 `lora/dora/qlora` 模式；`full` 模式会 fail-fast。
- 显存优化策略：
  - `value_model_mode=shared_backbone`：value 共享 policy backbone，不再 deepcopy。
  - `reward_model_mode=adapter_disabled_policy`：reward 走 policy 的 disable_adapter 路径，避免额外 reward backbone 拷贝。
- 若要扩展到真正多模态在线 rollout，需要单独扩展 PPO pipeline/collator 与 TRL rollout 适配。
- 新增策略时，优先扩展配置映射和数据适配层，不要重写训练核心。
- PPO 当前处于“暂停开发（非生产）”状态；未完成项与恢复门槛统一记录在 [docs/ppo_todo.md](ppo_todo.md)。

## 2. 扩展数据层（新增 source_type / 新样本格式）

### 2.1 必改文件
1. `src/shaft/data/dataset.py`（新增 Record / Dataset）
2. `src/shaft/data/sources.py`（新增 loader + data source 注册）
3. `src/shaft/data/center.py`（若改动影响多数据源装配 / mixing / online transform 编排）
4. `src/shaft/data/collator.py`（新增 collator）
5. `src/shaft/data/__init__.py`（导出）
6. `tests/test_data_sources.py`、`tests/test_data_center.py`、`tests/test_collator.py`

### 2.2 规则
- 新 source 必须使用注册器：`@register_data_source("xxx")`。
- 解析失败必须走聚合错误机制（输出失败行号和示例原因），不要只报第一条。
- Record 结构只描述“样本事实”，不要包含训练阶段状态。
- 多数据源加载、offline transform、sample-level mixing、dataset-aware online transform 的汇总入口是 `ShaftDataCenter`；不要把这些逻辑重新写回 pipeline。
- 如果扩展的是 mixing 规则或多源装配行为，优先修改 `src/shaft/data/center.py` / `src/shaft/data/mixing.py`，而不是在训练主流程里加分支。
- 若是“命名数据集”扩展，优先新增/维护 registry YAML，而不是把所有数据源都直接写进训练 YAML。
- `data.registry_path` 中的相对路径按 registry 文件目录解析；`data.datasets` 中的相对路径按主 config 文件目录解析。
- registry 解析发生在 `config.load_config()` 阶段，进入 pipeline 之前必须已经落成标准 `DataSourceConfig` 列表。

## 3. 扩展模型族（Qwen3VL 之外）

### 3.1 必改文件
1. `src/shaft/model/<model_family>.py`
2. `src/shaft/template/<model_family>.py`
3. `src/shaft/model/policies.py`（若新增 processor/peft policy）
4. `src/shaft/model/types.py`（必要时扩展 meta 字段 / `ShaftModelAdapter` 能力）
5. `tests/test_model_registry.py`、`tests/test_template_registry.py`

### 3.2 规则
- 模型专属实现必须显式带模型名，不使用泛名。
- 模型运行时入口统一是 `ShaftModelAdapter`，loader/collator/infer 不应该分别手写模板解析、processor policy 调度、target_modules 解析。
- `ModelMeta` 负责声明默认模型族元信息；`ShaftModelAdapter` 负责把模型名匹配结果与 group override 收敛成运行时单对象。
- `processor_policy` / `peft_policy` 优先通过 `src/shaft/model/policies.py` 注册后复用，再挂到 `ModelMeta` 或 `ModelGroup`。
- `ModelMeta` / `ModelGroup` 中明确：
  - `default_template` / `template`
  - `processor_policy`
  - `peft_policy`
  - `requires` / `additional_saved_files`
- group 级 override 只允许放“模型匹配后确实会变化”的能力，例如模板、pixel budget 支持、target_modules 策略，不要把 loader 逻辑复制到 group 配置里。
- checkpoint 兼容逻辑必须遵循 HF/PEFT 标准目录。

## 4. 续训与导出扩展

### 4.1 当前规则
- `init_from_checkpoint`：初始化权重（full 或 adapter）。
- `resume_from_checkpoint`：恢复 trainer 状态（需 `trainer_state.json`）。
- 模式校验：
  - full <-> full
  - lora/dora/qlora <-> adapter

### 4.2 新功能接入时
- 先补测试矩阵，再改实现：
  - `tests/test_checkpointing.py`
  - 必要时新增 `tests/test_<algo>_checkpointing.py`

## 5. 命令行扩展规则

- 统一入口：`scripts/train.py`
- 子命令注册：`src/shaft/cli/registry.py`
- 公共参数：`src/shaft/cli/common.py`
- 只允许无歧义覆盖参数（run-id/seed/epochs/lr/...）。
- 覆盖逻辑必须映射到结构化 config 字段，禁止隐式魔法行为。

## 6. 测试要求（提交前）

至少执行：
1. `pytest -q`
2. 新增能力对应测试文件单独执行一次（便于定位）

建议执行：
1. `pytest -q -m integration`（若改动影响真实模型加载/推理）

## 7. 常见越界反例

1. 在 `training` 里解析 `jsonl` 字段。  
2. 在 `data` 里写 optimizer/scheduler 分支。  
3. 在 `pipeline` 中硬编码某个模型族的 tokenizer 细节。  
4. 直接在 trainer 里拼装 prompt 模板字符串。  

发现以上反例，应把逻辑迁回对应层再提交。  
