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
   - 页面级筛选优先复用 `AdvancedFilterBar`，避免每页维护一套 search/filter 布局。
   - Overview 是总控工作台，只展示粗粒度运营信号、趋势和入口，不展示 recall 等细粒度模型指标。
   - 弹窗统一走 `WorkspaceDialog`；关闭按钮、Escape/backdrop 行为和 dialog body 滚动语义不能在业务页复制。
   - 标准按钮统一走 `ActionButton`、`CommandButton` 或 `IconActionButton`；业务页只保留样本行、
     label chip、画布 HUD 等专用交互控件。
2. **API Facade Layer**
   - FastAPI route、request/response 转换、错误响应和日志。
   - 不直接实现 metric、runtime lifecycle、prediction parsing。
   - 面向 agent 的稳定接口必须先落在 API/CLI，再由前端消费，不能要求 agent 读取前端状态或手改 store。
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
   - Benchmark copy、run manifest、editable run note、prediction snapshot、report、comparison、trash。
   - Artifact 是可追溯事实，不从 UI 临时状态回填语义。
   - `run.json` 是评测配置快照；`runs/<run_id>/note.json` 是可编辑备注真源，供人类和 agent 记录复现线索、idea 来源与排障细节。
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
- `projects/eval_bench/eval_bench/database.py`
  - 维护 job/service/prompt template registry；job/service 列表过滤和分页由 `job_page` /
    `service_page` 提供，CLI/API/Jobs/Services 页面不能各自复制高级检索语义。
- `projects/eval_bench/eval_bench/sample_paths.py`
  - 维护 sample image path 和 prediction JSON relative path 的命名与 fallback 规则。
- `projects/eval_bench/eval_bench/sample_scope.py`
  - 维护 run target label scope、实例过滤、payload 裁剪和 sample diagnostics 重映射。
- `projects/eval_bench/eval_bench/store.py`
  - 维护 benchmark/run 列表过滤、分页和 query 语义；CLI 与 `/api/benchmarks`、`/api/runs`
    必须复用同一份 store page 方法。
  - 维护 run note 的读写、列表摘要和长度校验；dashboard/API/CLI 不直接改 `note.json`。
  - 维护 rank board 的过滤、facet 计数、综合分公式、排序和分页输出；Compare 页不能再作为排行榜真源。
- `projects/eval_bench/eval_bench/comparison.py`
  - 维护 pairwise comparison 报告生成、历史摘要和 task/label/query 过滤；API/CLI/前端不各自复制
    comparison 历史检索语义。
- `projects/eval_bench/frontend/src/statusModel.ts`
  - 维护前端状态文案和操作启用条件。
- `projects/eval_bench/frontend/src/workspaceSettings.ts`
  - 维护 viewer 外观、交互和快捷键配置。
- `projects/eval_bench/frontend/src/filterControls.tsx`
  - 维护 `FilterSelect` 和 `AdvancedFilterBar`，是页面级高级检索控件真源。
- `projects/eval_bench/frontend/src/rankBoardPage.tsx`
  - 维护独立排行榜工作台页面；作为懒加载路由拆分，避免继续把核心工作台堆进 `main.tsx`。
- `projects/eval_bench/frontend/src/comparePage.tsx`
  - 维护成对 run 对比工作台页面；只展示 comparison 报告、历史对比和上下文入口，不再维护排行榜语义。
- `projects/eval_bench/frontend/src/ui.tsx`
  - 维护 `WorkspaceDialog`、`DataTable`、`Badge`、`ActionButton`、`CommandButton` 和
    `IconActionButton` 等基础展示组件；业务页不直接实现弹窗外壳或标准按钮层级。

## Extension Rules

- 新增任务类型：先加 schema task，再加 prompt template、parser、metric profile、viewer capability。
- 新增 metric：先加 `metric_profiles.py` registry，再在 `metrics/` 中实现 profile matcher 和聚合。
- 新增 target label scope：先加 prompt metadata 或显式 spec 字段，不允许通过 UI 默认值悄悄覆盖。
  前端可以提供 target labels 输入，但留空语义必须交给 `label_policy.py` 解析，不能在页面里复制
  layout/arrow 的默认 label policy。
- 新增 run annotation 字段：优先落在 run 目录的独立 artifact，再由 store 暴露读写接口；不要把可编辑备注写回不可变 run manifest。
- 新增 rank board 字段、facet 或排序规则：先更新 store/API/CLI 的 `rank-board` 输出，再同步前端表格和测试。
- 新增 agent 可操作对象：先提供稳定 CLI/API 查询入口；基础对象枚举应优先复用
  `list-benchmarks`、`list-runs`、`list-jobs`、`list-services`、`list-comparisons`，不要让 agent 读取前端状态或扫描 artifact 目录。
- 新增 job 入队入口：CLI 和 API 必须共享 `preflight_job_payload` / prompt template 解析；agent 先用
  `preflight-job` 校验，再用 `create-job` 入队，不能直接写 SQLite job record。
- 新增 job 状态：先更新 `job_lifecycle.py`，再更新 database、orchestrator、dashboard、status model 和测试。
- 新增 viewer 功能：先确定是 rendering capability 还是 command action；不能把功能混入 page 组件。
- 新增页面筛选：先复用 `AdvancedFilterBar`，后端已有稳定查询参数时再接 API；不要在页面内临时拼一套独立 search bar。Runs、Compare 和 Rank Board 的 run 过滤维度应优先保持一致。
- 新增页面标准动作：先复用 `ActionButton`、`CommandButton` 或 `IconActionButton`；只有样本行、画布
  HUD、label chip 等具有独立交互语义的控件才允许保留专用 button 样式。
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
- CLI/agent entrypoints: `projects/eval_bench/tests/test_cli.py`
- Frontend command/settings/viewer: frontend `test:shortcuts`、`test:workspace-settings`、
  `test:viewer-performance`、`test:layout` 和 render checks。
