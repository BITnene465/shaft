# Eval Bench Architecture

Eval Bench 是 Shaft 仓库内的离线评测工作台，不是训练主链的一部分。它的目标是把
benchmark copy、run snapshot、prediction import、metric report、comparison 和可视化检查做成
可追溯、可扩展的内部工程系统。

## Layer Model

Eval Bench 使用七层边界。新增功能必须先落在正确层级，再由上层组合。

1. **Presentation Layer**
   - React pages、workspace layout、dialog、table、viewer panel。
   - 只负责展示和用户操作编排，不推断 eval 语义。
   - 前端全局命令必须走 command/action registry，不能直接写死快捷键。
2. **API Facade Layer**
   - FastAPI route、request/response 转换、错误响应和日志。
   - 不直接实现 metric、runtime lifecycle、prediction parsing。
3. **Control and Lifecycle Layer**
   - Job registry、scheduler、resource lease、cancel request、service registry。
   - `job_lifecycle.py` 是 job 状态和调度资源占用规则的入口。
   - 运行中 job 被请求取消后，只要 worker/runtime 仍存活，就仍然占用调度资源。
4. **Execution Layer**
   - Worker、runtime adapter、OpenAI/vLLM client、process group cleanup。
   - 负责把 job manifest 执行为 raw output 和 prediction snapshot。
   - 不在这里决定 target label scope 或 metric profile。
5. **Evaluation Semantics Layer**
   - Prompt template、target label policy、metric profile、parser/profile 选择。
   - `eval_semantics.py` 是 evaluator/import/comparison 的语义入口。
   - `label_policy.py` 返回 label 集合和来源；`metric_profiles.py` 管理 profile registry。
   - `metrics/engine.py` 根据 profile 执行 matcher、样本诊断和聚合。
   - comparison 只比较持久化 report，必须保留 profile 的主指标语义。
   - 任何 task 子类型都必须通过 profile/label policy 表达，不允许只靠 `task=detection`。
6. **Artifact and Store Layer**
   - Benchmark copy、run manifest、prediction snapshot、report、comparison、trash。
   - Artifact 是可追溯事实，不从 UI 临时状态回填语义。
   - Store 可做索引、分页和读取优化，但不改变评估标准。
   - `sample_paths.py` 是 GT JSON、原图路径和 prediction JSON 路径映射的单一真源。
     worker、evaluator、prediction import、store 不允许各自复制 `.png/.jpg` fallback 规则。
   - `sample_scope.py` 是 run sample 按 `target_labels` 裁剪 GT、prediction payload 和 diagnostics
     的单一真源。
   - Benchmark sample 视图展示原始 benchmark 的全量 label；Run sample 视图必须按 run
     spec 的 `target_labels` 过滤 GT、prediction、label options 和实例计数。
7. **Rendering and Asset Layer**
   - Image proxy、preview、tile、viewer geometry、overlay color/style。
   - 只消费 sample detail 和 workspace settings，不参与 metric 计算。
   - 图片派生缓存和叠图渲染需要可测的性能边界。
   - 样本检查器只放高频审阅控件，例如 label、GT/pred 和几何层开关；overlay 数值样式、
     label 颜色和快捷键配置统一归入工作台设置页，避免在样本页形成第二套设置入口。

## Dependency Direction

允许方向：

```text
Presentation -> API Facade -> Control/Lifecycle -> Execution -> Artifact
Presentation -> Rendering/Asset -> Artifact/API payload
Execution -> Evaluation Semantics -> Metric/Profile registries
Evaluator/Comparison/Import -> Evaluation Semantics -> Artifact
```

禁止方向：

- Evaluation Semantics 不能导入 dashboard、worker 或 React 前端。
- Artifact/Store 不能推断 prompt label scope 或 metric profile。
- Presentation 不能拥有 job 状态机、metric matcher 或 service lifecycle 真源。
- Worker 不能按 prompt id 字符串自行判断 layout/arrow/keypoint。

## Current Middle-Layer Truth Sources

- `projects/eval_bench/eval_bench/eval_semantics.py`
  - 统一解析 run spec 的 `task`、`metric_profile`、`target_labels` 和 `target_labels_source`。
- `projects/eval_bench/eval_bench/metric_profiles.py`
  - 维护 `detection_iou_v1`、`keypoint_endpoint_v1` 等 metric profile。
- `projects/eval_bench/eval_bench/metrics/`
  - 维护 profile-driven matcher、geometry primitive、sample diagnostic 和 label aggregation。
  - `keypoint_endpoint_v1` 使用 ordered endpoint distance matcher，不再用 bbox IoU 决定 TP/FP/FN。
- `projects/eval_bench/eval_bench/label_policy.py`
  - 维护目标 label scope，返回 `explicit`、`prompt_metadata`、`legacy_prompt_id`、
    `task_default` 或 `unscoped` 来源。
- `projects/eval_bench/eval_bench/job_lifecycle.py`
  - 维护 job terminal/active/cancelled-resource lease 语义。
- `projects/eval_bench/eval_bench/sample_paths.py`
  - 维护 sample image path 和 prediction JSON relative path 的命名与 fallback 规则。
- `projects/eval_bench/eval_bench/sample_scope.py`
  - 维护 run target label scope、实例过滤、payload 裁剪和 sample diagnostics 重映射。
- `projects/eval_bench/frontend/src/statusModel.ts`
  - 维护前端状态文案和操作启用条件。
- `projects/eval_bench/frontend/src/workspaceSettings.ts`
  - 维护 viewer 外观、交互和快捷键配置。

## Extension Rules

- 新增任务类型：先加 schema task，再加 prompt template、parser、metric profile、viewer capability。
- 新增 metric：先加 `metric_profiles.py` registry，再在 `metrics/` 中实现 profile matcher 和聚合。
- 新增 target label scope：先加 prompt metadata 或显式 spec 字段，不允许通过 UI 默认值悄悄覆盖。
- 新增 job 状态：先更新 `job_lifecycle.py`，再更新 database、orchestrator、dashboard、status model 和测试。
- 新增 viewer 功能：先确定是 rendering capability 还是 command action；不能把功能混入 page 组件。
- 新增 sample 路径规则：只改 `sample_paths.py`，并用 store/worker/evaluator/import 的 focused
  测试证明四条调用链一致。
- 新增 run sample 展示范围规则：只改 `sample_scope.py`，不能在 dashboard route、viewer 或 store 中各自
  手写 label 过滤。

## Tests Required By Layer

- Evaluation semantics: `projects/eval_bench/tests/test_eval_semantics.py`
- Evaluator/report/comparison: `projects/eval_bench/tests/test_evaluator.py`
- Prediction import: `projects/eval_bench/tests/test_prediction_import.py`
- Job lifecycle/scheduler: `projects/eval_bench/tests/test_orchestrator.py`
- Dashboard API lifecycle: `projects/eval_bench/tests/test_dashboard.py`
- Frontend command/settings/viewer: frontend `test:shortcuts`、`test:workspace-settings`、
  `test:viewer-performance` 和 render checks。
