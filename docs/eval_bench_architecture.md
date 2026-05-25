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
   - 页面级筛选优先复用 `AdvancedFilterBar`；默认只露出 Filter 入口和可点击条件 token，
    展开后再以浮层显示检索表单，避免把多组 select 直接堆在主工作区；单条件清除、清空筛选、默认值判定和生效条件计数也只在
    `AdvancedFilterBar` 中维护。
   - Overview 是总控工作台，只展示粗粒度运营信号、趋势和入口，不展示 recall 等细粒度模型指标。
   - 弹窗统一走 `WorkspaceDialog`；关闭按钮、Escape/backdrop、body scroll lock、焦点进入、Tab 焦点闭环、
     关闭后焦点恢复和 dialog body 滚动语义不能在业务页复制。
   - 标准按钮统一走 `ActionButton`、`CommandButton`、`IconActionButton` 或 `PanelToggleButton`；
     业务页只保留样本行、label chip、画布 HUD 等专用交互控件。
   - 页面局部输入控件优先走 `controlPrimitives.tsx`；业务页不能为了局部 toolbar 继续复制
     `filter-select compact` 这类筛选样式。
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
   - `run.json` 是评测配置快照；`runs/<run_id>/note.json` 是可编辑备注真源，供人类和 agent 记录复现线索、idea 来源与排障细节；
     覆盖写必须支持基于 `updated_at` 的乐观并发校验，避免旧页面或旧 agent 上下文覆盖新备注。
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
    `task_default` 或 `unscoped` 来源；`legacy_prompt_id` 只能匹配内置 prompt 命名族
    `grounding_layout.*`、`grounding_arrow.*`、`keypoint_arrow.*` 和历史 `arrow_keypoint.*`。
- `projects/eval_bench/eval_bench/job_lifecycle.py`
  - 维护 job terminal/active/cancelled-resource lease 语义。
- `projects/eval_bench/eval_bench/database.py`
  - 维护 job/service/prompt template registry；job/service 列表过滤和分页由 `job_page` /
    `service_page` 提供，CLI/API/Jobs/Services 页面不能各自复制高级检索语义。
  - `matching_jobs()` 是 job lifecycle 和 scheduler 的完整匹配集合真源；`job_page()` / `list_jobs()`
    只负责分页展示，不能用于 running resource check 或 queued FIFO scan。
- `projects/eval_bench/eval_bench/sample_paths.py`
  - 维护 sample image path 和 prediction JSON relative path 的命名与 fallback 规则。
- `projects/eval_bench/eval_bench/sample_scope.py`
  - 维护 run target label scope、实例过滤、payload 裁剪和 sample diagnostics 重映射。
- `projects/eval_bench/eval_bench/store.py`
  - 维护 benchmark/run 列表过滤、分页和 query 语义；CLI 与 `/api/benchmarks`、`/api/runs`
    必须复用同一份 store page 方法；Benchmarks/Runs 页面只能提交查询条件，不能复制高级检索语义。
  - 维护 benchmark summary 的 `labels` 输出；manifest 缺少 labels 时可通过 sample scan fallback 补齐。
    任务创建 UI 和 agent CLI 只能消费这个 store 字段，不能自行扫描 benchmark artifact。
  - 维护 run note 的读、覆盖写、追加写、列表摘要和长度校验；dashboard/API/CLI 复用 store，
    不直接改 `note.json`。
  - 维护 rank board 的过滤、facet 计数、F1 默认主指标、非加权主指标切换、列表排序维度和分页输出；
    Compare 页不能再作为排行榜真源。`score` 在非加权模式下镜像当前主指标，显式 weighted scheme
    下才代表加权分。
- `projects/eval_bench/eval_bench/comparison.py`
  - 维护 pairwise comparison 报告生成、已保存 report 读取、成对样本详情 payload、历史摘要和
    task/label/query 过滤；API/CLI/前端不各自复制 comparison 历史检索或样本 payload 语义。
- `projects/eval_bench/frontend/src/statusModel.ts`
  - 维护前端状态文案和操作启用条件。
- `projects/eval_bench/frontend/src/workspaceSettings.ts`
  - 维护 viewer 外观、交互、快捷键、图层显示和 label 选择等浏览器本地偏好。
- `projects/eval_bench/frontend/src/filterControls.tsx`
  - 维护 `FilterSelect` 和 `AdvancedFilterBar`，是页面级高级检索触发器、浮层表单、分组目录、条件 token、清空动作和默认值判定真源。
- `projects/eval_bench/frontend/src/controlPrimitives.tsx`
  - 维护 number、color、select、toggle 等局部输入基础控件；manifest toolbar、viewer、settings、
    弹窗表单和对比选择轨不各自复制 select/input 外壳。
- `projects/eval_bench/eval_bench/log_utils.py`
  - 维护 backend log、job runtime log tail 和 job log path 解析；Dashboard API 和 CLI 共用，不在两端各自拼路径。
- `projects/eval_bench/frontend/src/overviewPage.tsx`
  - 维护总控工作台页面；作为独立路由模块承载下一步动作、评测管线、核心运行态、readiness switchboard 和最近 run 摘要。
- `projects/eval_bench/frontend/src/benchmarksPage.tsx`
  - 维护基准集目录、创建副本弹窗和基准集真值检查器；作为懒加载路由拆分，避免检查器逻辑回流 `main.tsx`。
- `projects/eval_bench/frontend/src/samplePager.tsx`
  - 维护 benchmark/run 检查器共享样本分页控件、目录分页 `PagerControl` 和通用 offset clamp；
    Runs、Benchmarks、Jobs、Services、Compare 和 Rank Board 只能传 className/meta，不各自复制上一页/下一页逻辑。
- `projects/eval_bench/frontend/src/runsPage.tsx`
  - 维护结果库、导入预测弹窗、带结构化模板插入的 run note 编辑器和 run 样本检查器；作为懒加载路由拆分。
- `projects/eval_bench/frontend/src/labelSubtaskControls.tsx`
  - 维护 detection label 子任务 chip、默认策略和全部候选选择；Jobs 和 Runs import prediction
    只能复用该组件，不暴露自由文本 label 追加路径，keypoint 不暴露 label 子任务 UI。
- `projects/eval_bench/frontend/src/sampleViewer.tsx`
  - 维护 Run Inspector 与成对样本对比共享的 GT / Prediction 叠图、对象检查器和 viewer 偏好状态。
- `projects/eval_bench/frontend/src/rankBoardPage.tsx`
  - 维护独立排行榜工作台页面；作为懒加载路由拆分，承载可见的主指标切换、升降序、Top contenders、score spread、
    facet rail、weighted scheme 和分页表格，避免继续把核心工作台堆进 `main.tsx` 或藏进 Compare。
- `projects/eval_bench/frontend/src/comparePage.tsx`
  - 维护成对 run 对比工作台页面；只展示 comparison 报告、历史对比和上下文入口，不再维护排行榜语义。
- `projects/eval_bench/frontend/src/comparisonSamplePage.tsx`
  - 维护成对样本对比详情；作为懒加载路由拆分，复用 `sampleViewer.tsx` 而不维护第二套叠图。
- `projects/eval_bench/frontend/src/settingsPage.tsx`
  - 维护工作台设置页面；`main.tsx` 只负责路由到该页面，不承载设置页预览、分组或本地偏好编排。
- `projects/eval_bench/frontend/src/ui.tsx`
  - 维护 `WorkspaceDialog`、`DataTable`、`Badge`、`ActionButton`、`CommandButton`、
    `IconActionButton`、`IconNavLink`、`InlineNavLink`、`InlineAnchor` 和 `PanelToggleButton` 等基础展示组件；样本行选择使用 `SelectableRowButton`，query/label chip
    使用 `OptionChipButton`，可选卡片使用 `SelectableCardButton`；表单提交和 Settings 快捷键捕获
    也直接使用 `ActionButton` 变体，不保留页面私有 submit/capture raw button；
    业务页不直接实现弹窗外壳、标准按钮层级、图标/文本导航链接或重复的 row/chip button 形态。
  - `WorkspaceDialog` 是弹窗焦点、滚动锁和可访问性属性的单一真源；业务页只传 title/meta/content，不直接操作
    body overflow 或手写焦点陷阱。
- `projects/eval_bench/frontend/src/controlPrimitives.tsx`
  - 维护紧凑 select、表单 select、数值输入、颜色输入和开关等基础控制原语；viewer 图层预设、
    settings 控件、manifest toolbar、Runs/Services 弹窗和 Compare 选择轨不直接手写同类 select 外壳。

## Extension Rules

- 新增任务类型：先加 schema task，再加 prompt template、parser、metric profile、viewer capability。
- 新增 metric：先加 `metric_profiles.py` registry，再在 `metrics/` 中实现 profile matcher 和聚合。
- 新增 target label scope：先加 prompt metadata 或显式 spec 字段，不允许通过 UI 默认值悄悄覆盖。
  前端可以提供 target labels 输入，但留空语义必须交给 `label_policy.py` 解析，不能在页面里复制
  layout/arrow 的默认 label policy。agent 查询 target label 作用域必须走 `resolve-target-labels`，
  不能直接扫描 benchmark artifact 或 prompt registry。
- 新增 detection label 子任务 UI：必须通过 manifest 或 import payload 的 `target_labels` 修改显式 spec，
  候选 label 来自 benchmark summary / prompt template / 当前 manifest，不能在页面层硬编码任务 label，也不能
  在共享面板里暴露自由文本 label 添加入口。`preflight-job`
  必须在 benchmark label index 存在时拒绝未知 `target_labels`，避免拼错 label 的 job 被 agent 或 UI 入队。
  前端应用 prompt template 后必须重新按 manifest task 选择兼容 benchmark；不能只因为旧 benchmark id
  仍存在就保留一个 task 不匹配的 job draft。
  Keypoint 不暴露 label 子任务 UI；`resolve-target-labels` 必须返回 `label_subtasks_supported=false`，
  只保留默认 arrow 关键点评估范围。`label_policy.py` 是这条边界的后端真源，preflight、init-run、
  prediction import、worker 和 evaluator 都必须拒绝 keypoint 上非 `arrow` 的显式 `target_labels`。
- 新增 run annotation 字段：优先落在 run 目录的独立 artifact，再由 store 暴露读写接口；不要把可编辑备注写回不可变 run manifest。
  覆盖类写入必须支持 expected `updated_at` 保护；agent 增量线索优先使用 append 入口。
- 新增 rank board 字段、facet、排序规则或分页状态：先更新 store/API/CLI 的 `rank-board` 输出，再同步前端表格、分页控件和测试；前端不能退回固定首屏 slice。
  表格第一分数列必须消费后端 `primary_metric_label` 和 entry `score`，不能在主指标切换后仍固定展示
  F1 作为排名依据。
  前端 facet rail 必须以 API 返回的 Tasks、Benchmarks、Status、Labels、Models、Prompts 和 Metrics
  为真源，直接驱动同一份高级检索状态，不能维护一套只展示 facet 计数的静态 UI；长 facet 组默认可折叠，
  但必须提供展开/收起入口，不能因为首屏密度只渲染前 5 项而丢掉长尾筛选值。
- 新增 agent 可操作对象：先提供稳定 CLI/API 查询入口；基础对象枚举应优先复用
  `list-job-templates`、`show-job-template`、`list-prompt-templates`、`show-prompt-template`、
  `init-run`、`validate-prediction`、`list-benchmarks`、`show-benchmark`、`list-runs`、
  `show-agent-command`、`list-jobs`、`show-job`、`process-next-job`、`list-services`、`show-service`、
  `list-comparisons`、`show-comparison`、`show-comparison-sample`，不要让 agent
  读取前端状态、SQLite 或扫描 artifact 目录。CLI parser 暴露的每个子命令必须登记到 `_command_handlers()`，
  agent 稳定命令必须登记到 `AGENT_COMMAND_METADATA` 并可由 `list-agent-commands` 发现；metadata 需要声明
  `domain` 和 `mutates_state`，并能通过 `show-agent-command --name <command>` 读取单条命令契约；删除、归档、取消、停止这类危险生命周期命令还必须进入
  `AGENT_DESTRUCTIVE_COMMANDS` 并在命令发现输出中标记 `destructive`；命令发现输出还必须包含顶层 `recommended_runner`、每条命令的
  `argv_prefix`、稳定单行 `usage`，参数 schema 从 argparse parser 自动导出为 `arguments` 和
  `mutually_exclusive_groups`，`AGENT_STABLE_COMMANDS` 由 metadata 派生。这些集合由
  `test_cli_parser_commands_have_handlers_for_agent_contract` 锁住，避免新增命令只加 parser 或只加 handler，
  或者缺少 agent 判断副作用和参数形态所需的元信息。
- 新增 CLI 命令或 dashboard route：模块顶层只能保留轻量依赖。`dashboard`、`worker`、`evaluator`、
  Shaft/Transformers 这类重运行时必须在具体命令或 route 内懒加载，保证 `list-*`、`rank-board`、
  run note 等 agent-safe 入口可以快速 import。
- 新增或改动 agent 关键命令：`show-agent-command` / `list-agent-commands` 必须同步暴露可执行参数、
  互斥组、副作用标记和必要的 `output_schema`。Rank Board 这类核心只读命令必须描述分页、filters、
  facets、primary metric 和 entry 字段；run note 与 label policy 命令必须描述 note/concurrency 字段和
  detection/keypoint label 子任务字段；run/sample inspection 命令必须描述 summary、payload、diagnostics
  和 scoped label 字段；job/service/comparison 查询命令必须描述 record、runtime、delta 和成对样本详情字段，
  避免 agent 通过猜测 JSON 字段或读取 store 内部结构完成任务。
- 新增 prompt template 管理能力：API 与 CLI 必须共用 `EvalBenchDatabase` 的 registry；前端只能消费
  同一 registry，不能在页面里维护独立 prompt template 列表。
- 新增 job 入队入口：CLI 和 API 必须共享 `preflight_job_payload` / prompt template 解析；agent 先用
  `preflight-job` 校验，再用 `create-job` 入队，不能直接写 SQLite job record。`create-job` 和
  `/api/jobs` 必须持久化非阻塞 `preflight_warnings`，避免 agent 只能从一次性 stdout 中看到风险提示。
- 新增 agent 生命周期入口：CLI 必须优先复用 `EvalBenchStore`、`EvalBenchDatabase`、`EvalBenchServiceManager`
  和 `log_utils.py`；不能让 agent 直接改 run manifest、SQLite、service 目录或 runtime log 路径。
- 新增 job 状态：先更新 `job_lifecycle.py`，再更新 database、orchestrator、dashboard、status model 和测试。
- 新增 job 调度或 worker claim 入口：资源占用检查、running job 检查和 queued FIFO scan 必须使用
  `EvalBenchDatabase.matching_jobs()`，不能复用 UI 列表的固定 `limit` 窗口。
- 新增 viewer 功能：先确定是 rendering capability 还是 command action；不能把功能混入 page 组件。
- 新增页面筛选：先复用 `AdvancedFilterBar`，后端已有稳定查询参数时再接 API；不要在页面内临时拼一套独立 search bar。Runs、Compare 和 Rank Board 的 run 过滤维度应优先保持一致。Benchmark Inspector / Run Inspector
  的样本级 label、error 筛选也使用同一个折叠式筛选组件，避免侧栏堆叠 select。
  Runs、Compare、Benchmarks、Jobs、Services 和 Rank Board 这类结果列表必须走后端分页，筛选变化重置 offset，数据减少时 clamp 到最后有效页；
  Compare 翻页时必须保留已选 baseline/candidate run id，不能用当前页是否包含该 run 来清空 URL 状态。
  不能用固定 `limit=200` 的首屏 slice 代替完整结果浏览。
- 新增总览运行态信号：只能消费 store、job、service、scheduler 这些现有 API/CLI 真源；总览页保持粗粒度总控视角，
  不能重新展示 precision、recall、mIoU 等精细评测指标。
- 新增总览视觉模块：优先用 priority stage、hero next action、四个可行动信号和 pipeline progress rail 服务“当前是否可用、
  卡在哪里、下一步去哪”的判断，不再把状态分布拆成低价值 mini chart wall、活动矩阵或 Run/Ops/Volume 面板组；总览主体保持
  v9 mission-control surface：顶部两列只放 priority stage 和实时 command rail，底部只放一个 operations surface 与最近 run 紧凑 ticker。
  顶部 priority stage 只能承载当前系统态、同步状态、关键规模、benchmark -> run -> report -> rank board 流线和当前优先动作，
  不能回流二级诊断，也不能使用只表达装饰关系的 orbit 图。右侧 command rail 固定展示覆盖、
  待评、队列和服务四个可点击入口，并以 2x2 信号板表达实时状态；不再单独常驻阻塞优先级面板，卡点应体现在当前主动作和可点击状态入口中。readiness switchboard 固定聚合
  service、queue、evaluation 和 rank board 四个入口，每个入口展示状态、占比轨道和目标路由；最近 run
  必须按 `created_at` 倒序截取，且只展示 benchmark/model 与 prediction/report 数量，不能依赖 API 返回顺序。compact / narrow 视口允许页面滚动，但不能把
  focus、readiness 或 recent 核心面板压缩成不可读的折叠外壳。
  Parser、配置快照、artifact 明细、备注新鲜度、任务类型、模型分布、label footprint、样本/label 权重、
  Job 日历、scheduler 资源和推理参数桶这类低频排障信息不进入总览，留在 Runs / Inspector / Rank Board / Services。
  compact / narrow 视口需要滚动时由 Overview 页面栈承担，command desk 不应把核心面板裁成独立折叠容器；
  最近 run 只保留可点击紧凑摘要，不承载二级诊断面板。
- 新增 dashboard 交互动效：hover、pulse、rail transition 和入场动画只用于状态反馈、可点击性和实时感；
  不允许用大面积装饰动画替代信息结构，也不能让动效改变数据语义或造成滚动/布局抖动。
- 新增 dashboard 通用交互反馈：优先扩展共享按钮、卡片、表格行、chip、导航和状态胶囊的 hover/focus/active
  反馈，不要在单个业务页复制私有动效。
- 顶栏 profile/status 是独立 capsule，不再使用外层圆角容器；在线、同步中和异常的动效只落在 status pill
  本身，避免 wrapper 承担状态语义。
- 新增 dashboard icon：先在 `iconLibrary.tsx` 定义语义 key，再替换调用点；排行榜入口、入榜状态、已评估状态
  这类不同 UI 语义不能复用同一个通用 metrics icon。
- 新增加权排行：必须作为显式 `rank_scheme` / `rank_profile` 配置进入 store/API/CLI，权重项至少包含
  `benchmark_id`、`metric`、`weight` 和缺失指标处理规则；前端必须标明当前使用的是 weighted scheme。
  默认 Rank Board 始终以 `f1_iou50` 作为主指标；用户切换到 precision、recall、mIoU 或预测数时，
  store/API/CLI 必须同步更新 `primary_metric`、`primary_metric_label` 和 entry `score`。不能把加权结果
  重新命名成默认分数；显式 weighted
  scheme 的输出必须包含 `weighted_score`、原始 `rank_scheme` 和 entry-level `score_components`。
  Rank Board entry 必须同时输出 `score_delta`，以当前完整排序的第一名为基准计算主分数差值；分页后的
  entry 不能改用当前页第一名作为基准，避免人类和 agent 误判 leader gap。
  前端 Rank Board 只能把 weighted scheme 作为折叠式显式面板传给 `/api/rank-board`，不能在浏览器端另写
  一套加权计算；后端拒绝 scheme 时必须在面板内显示错误，不允许整页退化为加载失败。
- 新增页面标准动作：先复用 `ActionButton`、`CommandButton`、`IconActionButton` 或 `PanelToggleButton`；只有画布
  HUD 这类低层交互允许保留专用样式。样本行必须通过
  `SelectableRowButton` 维护 selected / aria-current 语义，query/label chip 必须通过
  `OptionChipButton` 维护 active / aria-pressed 语义，Compare 这类可选卡片和 viewer object row 必须通过
  `SelectableCardButton` 维护 active / aria-pressed 语义，局部 select 必须通过 `controlPrimitives.tsx`
  的 `CompactSelectControl` 或 `FormSelectControl`，避免业务页重复拼 className 或 raw `<select>`。
  Settings 快捷键捕获可以保留 `shortcut-capture` 专用样式和局部 `onKeyDown` 语义，但底层按钮仍必须通过
  `ActionButton`。
  前端 `test:ui-contracts` 是这条边界的静态防线，必须覆盖阻塞式浏览器弹窗、业务页自建 dialog shell
  和已收敛标准动作、row/chip/select 原语回流。
- 新增 sample 路径规则：只改 `sample_paths.py`，并用 store/worker/evaluator/import 的 focused
  测试证明四条调用链一致。
- 新增 run sample 展示范围规则：只改 `sample_scope.py`，不能在 dashboard route、viewer 或 store 中各自
  手写 label 过滤。
- 新增 inspector 样本过滤：过滤后 0 命中仍必须保留 `inspector-sidebar`、`AdvancedFilterBar` 和
  `viewer-panel`，只在样本列表/画布区域显示空结果；不能回退为全页 EmptyState 导致用户无法撤销过滤。
- 新增 inspector / viewer 响应式布局：桌面优先保持工作区一屏内分栏；compact / narrow 视口下 split
  堆叠时，`.visual-inspector-page` 必须允许局部滚动，`.image-stage` 必须保留可操作高度，对象检查器继续
  在自己的面板内滚动；不能用外层 `overflow: hidden` 把画布压成不可见。
- 新增目录分页页面：只允许复用 `PagerControl` 和 `clampListPageOffset`；页面可以定义 page size、
  API filter 和业务 className，但不能复制 pager range、上一页/下一页禁用和 offset clamp 逻辑。
- 新增可视化检查器主视图统计：外层 `VisibleMetricStrip` 只展示真实/预测数量；TP、FP、FN、IoU、P/R
  等精细指标必须留在折叠式分 label 明细、对象诊断、排行榜或对比页中。

## Tests Required By Layer

- Evaluation semantics: `projects/eval_bench/tests/test_eval_semantics.py`
- Evaluator/report/comparison: `projects/eval_bench/tests/test_evaluator.py`
- Prediction import: `projects/eval_bench/tests/test_prediction_import.py`
- Job lifecycle/scheduler: `projects/eval_bench/tests/test_orchestrator.py`
- Dashboard API lifecycle: `projects/eval_bench/tests/test_dashboard.py`
- CLI/agent entrypoints: `projects/eval_bench/tests/test_cli.py`
- Frontend command/settings/viewer: frontend `test:shortcuts`、`test:workspace-settings`、
  `test:viewer-performance`、`test:layout` 和 render checks。
