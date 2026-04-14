# Shaft 项目 Skill 规划

该文档定义团队协作时的“做什么、怎么做、怎么验收”。

## Skill 1：扩展模型族（以 `glm4v` 为例）
目标：在不改动训练核心逻辑前提下，支持新模型类型。

执行步骤：
1. 在 `src/shaft/model/` 新增 `glm4v.py`：
   - 新建 `ModelMeta`（`model_type="glm4v"`）
   - 实现 `ModelLoader`（加载模型、tokenizer、processor、template）
2. 在 `src/shaft/template/` 新增 `glm4v.py`：
   - 注册 `TemplateMeta` + `Template` 实现
3. 在 `src/shaft/model/types.py` 如有必要补齐模型特有 `ModelCapabilities` / `PeftPolicy`。
4. 在数据配置中增加示例并跑单测。

验收：
- `tests/test_model_registry.py` 能通过 `build_model_meta("glm4v")`。
- `scripts/train.py --config ... --run-id` 可完整初始化（`--max_steps` 小值 smoke）。

## Skill 2：引入新算法（DPO/PPO）
目标：把算法接入统一 pipeline 与 CLI，不污染 SFT 数据链路。

执行步骤：
1. 新增 `src/shaft/algorithms/<algo>.py` + `@register_algorithm("<algo>")`。
2. 在 `Algorithm.build_trainer` 中返回专用 Trainer/TRL Trainer 封装。
3. 若为 CLI 命令新增：
   - 更新 `src/shaft/cli/<algo>.py` 或复用 `rlhf.py`。
4. 补足 `schema` 配置项与参数映射。
5. 写 `tests/test_<algo>_algorithm.py`。

验收：
- 命令可被 `src/shaft/cli` 注册器发现。
- pipeline 能按配置选择并构建 `<algo>` 算法实例。

## Skill 3：多数据源与混合策略优化
目标：在一个训练作业里稳定接入多个任务数据源。

执行步骤：
1. 配置多个 `data.datasets`（每个 `train_paths/val_paths` 可多文件）。
2. 调整 `data.datasets[].weight` 与 `data.mix_strategy`。
3. 将字段级标准化前移到 `data/sources.py`。
4. 如需新增强方式，在 `data/transforms.py` 或新的 transform 模块注册。

验收：
- `tests/test_mixing.py` 覆盖 `interleave_under/interleave_over`。
- `tests/test_data_sources.py` 覆盖新源加载路径与 `dataset_id` 归属。

## Skill 4：日志与可观测性增强
目标：可追踪训练时间点与保存点。

执行步骤：
1. 使用 `@interceptor("pipeline.train.run", phase=...)` 包裹流程关键节点。
2. 使用 `@hook("before_step")` / `@hook("on_save")` 打日志与事件埋点。
3. 新增输出事件放入 `observability` 统一通道。

验收：
- 关键阶段有开始/结束事件；
- 训练/保存时 barrier 行为一致。

## Skill 5：infer 多阶段编排
目标：支持“视觉预处理 -> 结构化提取 -> 校验修复”串式推理。

执行步骤：
1. 在 `src/shaft/infer/schema.py` 补充阶段配置模型。
2. 配置多个 engine 与 stage（`InferPipelineConfig`）。
3. 在 `InferPipeline.run` 中设置上下文键名（`output_key`）和用户提示拼接。

验收：
- 一条 `json` 配置能完成两阶段推理并在 context 中返回 `__trace__`。

## 交付规范（所有 Skill 共用）
- 不修改非目标层职责。
- 每个新能力必须有：
  - 注册条目（registry + 装饰器）
  - 配置路径（`schema.py`）
  - 至少 1 个单测（pytest）
  - 文档更新到本文件对应 skill 或 `extension_guide.md`

## 未来扩展优先级
1. 完成 DPO baseline（第一阶段）
2. 完成 PPO baseline（第二阶段）
3. 扩展模型族注册（第三阶段）
4. 数据在线增强插件化（第四阶段）

