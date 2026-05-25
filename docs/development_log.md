# Shaft 开发日志

本文档记录已经暴露过的工程问题、指标误判、修复方式和后续防线。目标是让重复犯错的问题进入仓库真源，而不是停留在单次聊天或临时排障记录里。

## 维护规则

- 当线上/离线 eval 指标异常、训练语义被误判、或者同类 bug 第二次出现时，必须补一条日志。
- 每条日志至少包含：现象、根因、影响范围、修复、回归测试、后续防线。
- 如果问题涉及评估标准，必须明确区分“模型能力问题”和“eval/codec/metric 误判”。
- 日志不是待办列表；待实现事项可以同步到 `docs/todo.md`，但根因和经验必须留在这里。

## 2026-05-25: Eval Bench target-label CLI 文案误导 keypoint 子任务语义

### 现象

`label_policy.py` 已经明确 keypoint 任务固定评价 `arrow`，并会拒绝非 `arrow` 的显式 `target_labels`；
但 `init-run --target-label` 的 help 仍写成 “Limit detection/keypoint evaluation”，agent command contract
也会暴露这段文案，容易让 agent 误以为 keypoint 可以像 detection 一样选择任意 label 子集。

### 根因

label 子任务能力先在解析与验证层收口，CLI 参数说明没有同步从“通用 target label”改为“detection label subtask scope”。
agent 契约直接从 argparse action 抽取 help，因此 CLI help 的模糊描述会进入稳定 agent 命令面。

### 影响范围

- 影响 `init-run`、`import-predictions` 和 `resolve-target-labels` 的 agent-facing 参数契约。
- 不改变 evaluator、codec、metric、rank-board 或数据语义；这是 CLI/agent contract 文案偏差，不是模型能力问题，也不是评估标准误判。

### 修复方式

- 新增共享 `DETECTION_TARGET_LABEL_HELP`，统一三个 `--target-label` 参数说明。
- 文案明确 `--target-label` 代表 detection label 子任务范围，keypoint run 固定 `arrow` 且会拒绝非 `arrow` label。
- CLI contract 测试锁住这段 help，避免后续重新把 keypoint 描述成任意 label 子任务。
- README 同步 agent/CLI 使用说明。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_cli.py::test_cli_parser_commands_have_handlers_for_agent_contract projects/eval_bench/tests/test_cli.py::test_cli_resolves_target_labels_for_agent_label_subtasks`

### 后续防线

- 面向 agent 的 CLI 参数 help 必须和后端 policy 保持同一语义；新增或修改 agent 稳定命令时要同时检查 argparse help 和 policy 测试。

## 2026-05-25: Eval Bench 对比样本页 compact 视口横向裁切

### 现象

`test:layout` 在 compact 视口进入成对样本对比页时发现 `.comparison-run-panel` 的 `scrollWidth` 大于
`clientWidth`，但 pane 的 `overflow-x` 是 hidden，导致 run id 和“打开 run”入口组合在窄 pane 中可能被横向裁切。

### 根因

`comparison-run-heading` 是 flex 行，但标题容器没有 `min-width: 0`，长 run id 会按内容宽度参与布局，挤压右侧链接后扩大
panel 的横向 scroll size。外层 pane 为了避免 viewer 溢出设置了 hidden，因此问题表现为裁切而不是可滚动。

### 影响范围

- 影响 Compare Sample 页面 compact / narrow 视口下的可读性和滚动稳定性。
- 不改变 eval、codec、metric、rank-board、job、service 或数据语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- 给 `.comparison-run-heading` 及其标题容器补 `min-width: 0`。
- 将 run id 标题 `max-width` 约束到当前容器，并继续使用 `overflow-wrap: anywhere`，让长 id 在 pane 内换行。
- 在 `.comparison-run-panel` 嵌入场景隐藏 SampleViewer toolbar 的 legend/control 行；该 panel 已隐藏 side inspector，
  “收起检查器”和 FN/FP legend 在这里没有操作价值，继续显示只会撑开窄 pane。

### 回归测试

- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- split pane 内的 heading / toolbar 必须显式处理 `min-width: 0`；layout smoke 的横向裁切检查继续覆盖 compact 对比页。

## 2026-05-25: Eval Bench Overview 旧设计 CSS 轨道未清理

### 现象

总览页运行时代码已经切到 `overview-home-v8`，但 `styles.css` 仍保留 v2-v7 的 Overview 样式轨道，包括
`overview-command-deck`、`overview-home-v6`、`overview-home-v7`、旧活动矩阵和 orbit 相关样式。页面能正常显示，
但 CSS 真源同时描述多套互斥设计，后续维护时很容易把旧结构当成可复用状态。

### 根因

前几轮为了快速迭代首页视觉，把新样式追加到文件末尾，并依靠更高优先级 selector 覆盖旧样式；没有在结构稳定后做样式级收口。
UI contract 只检查 Overview 源码不回流旧组件，没有检查 stylesheet 是否还保留旧设计轨道。

### 影响范围

- 影响 Dashboard 总览页样式维护边界和静态契约可信度。
- 不改变 eval、codec、metric、rank-board、job、service 或数据语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- 删除 v2-v7 Overview CSS 轨道，只保留 `overview-home-v8` decision surface 和当前动画 keyframes。
- `test:ui-contracts` 增加 stylesheet 防线，禁止 `overview-home-v6`、`overview-home-v7`、`overview-command-deck`、
  `overview-command-center-redesign`、`overview-focus-panel`、`overview-side-stack`、`overview-right-rail`、
  `overview-activity-matrix`、`overview-chart-matrix` 和 `overview-mini-chart` 回流。
- README 和架构文档同步 Overview 模块描述，不再把独立阻塞优先级面板列为页面职责。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- Overview 后续视觉迭代必须替换当前轨道，而不是继续追加互斥版本；静态契约同时检查 TSX 和 CSS 两个真源。

## 2026-05-25: Eval Bench 首页 v7 仍像低价值面板堆叠

### 现象

总览页已经从旧图表墙收敛成 home cockpit，但 v7 仍把 orbit 装饰、流程面板、行动入口和最近 run 分成多块陈列。
用户视角下首屏没有形成“现在该做什么”的强主线，页面虽然压缩了信息密度，但仍像状态卡堆叠，缺少可感知的交互层级。

### 根因

上一轮只约束了“哪些低价值面板不能回流”，没有继续约束首页只能服务三个决策问题：当前是否可用、管线卡在哪里、
下一步去哪。`OverviewHeroMap` 只表达装饰性的四节点关系，和下方真实 pipeline 重复；readiness panel 又套在
右侧 rail 中，导致空间层级被组件容器而不是操作价值决定。

### 影响范围

- 影响 Dashboard 总览页的信息价值、空间分配和 hover/动效反馈。
- 不改变 eval、codec、metric、rank-board、job、service 或数据语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- 总览升级为 `overview-home-v8` decision surface：顶部两列只保留 priority stage 和 command rail，底部只保留
  operations surface 与最近 run 摘要。
- 删除 `OverviewHeroMap` / orbit 装饰，把四段 benchmark -> prediction -> report -> rank board 流线直接放进
  priority stage，避免重复展示流程。
- readiness switchboard 直接成为 operations surface 的主体，不再额外套 `overview-action-panel` 和右侧 rail。
- 增加 `overview-decision-strip`、`overview-impact-strip`、pipeline stage hover、signal hover、action hover 和同步 pill
  动效，让交互感来自可点击入口和实时状态，而不是装饰动画。
- README、架构文档、脚本文档、UI contract 和 layout smoke 同步 v8 结构边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- Overview 新增模块必须先证明服务“当前是否可用 / 管线卡在哪里 / 下一步去哪”之一；不能新增只占空间的诊断面板或装饰关系图。
- 首页动效只能绑定在同步状态、可点击入口、进度轨和 hover 反馈上，不能改变数据语义或造成布局抖动。

## 2026-05-25: Eval Bench 表格图标导航链接仍手写 icon-button

### 现象

Dashboard 已经把按钮、chip、card 和弹窗交互大多收敛到 `ui.tsx`，但 Runs 表格中的样本检查入口仍直接在
业务页写 `Link className="icon-button dense"`。这个入口本质是标准图标导航链接，却没有共享 `title`、
`aria-label`、dense class 和 icon-button 样式拼接逻辑。
同类问题也存在于若干 router 和动态 href 文本入口：页面直接写 `Link className="mini-link ..."` 或
`a className="mini-link ..."`，导致 mini-link 变体继续在业务页分散拼接。

### 根因

前期只给可触发动作的图标按钮建立了 `IconActionButton`，没有给 link 形态的图标入口、router mini-link
和 href mini-link 入口建立对应原语。因此表格行 action 区和多个页面 command link 仍保留页面私有的链接样式路径。

### 影响范围

- 影响 run 表格行操作区的按钮/链接组件边界和可访问性一致性。
- 不改变 API、run note、rank-board、eval metric、viewer 几何或数据语义。

### 修复方式

- `ui.tsx` 新增 `IconNavLink`，统一 router 图标链接的 `icon-button`、dense、`title` 和 `aria-label` 语义。
- `ui.tsx` 新增 `InlineNavLink`，统一 router mini-link 的基础 class、icon 和 children 拼装。
- `ui.tsx` 新增 `InlineAnchor`，统一动态 href mini-link 的基础 class、icon 和 children 拼装。
- Runs 表格样本检查入口改用 `IconNavLink`；Benchmark、Jobs、Rank Board 和 Compare 的 router mini-link
  入口改用 `InlineNavLink`；Compare 的动态样本入口和成对样本页的 run 入口改用 `InlineAnchor`。
- UI contract 增加防线，禁止 run table row icon link 回退为业务页手写 `icon-button dense`，并禁止 router
  / href mini-link 回退为页面私有 class。
- README 和架构文档同步按钮/链接原语边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`

### 后续防线

- 新增 router 图标入口优先复用 `IconNavLink`，新增 router mini-link 入口优先复用 `InlineNavLink`，
  新增 href mini-link 入口优先复用 `InlineAnchor`；只有普通正文链接或明确的业务导航卡片可以直接使用
  `Link` / `<a>` 样式。

## 2026-05-25: Eval Bench run note 覆盖保存缺少版本保护

### 现象

Run note 已经有 Dashboard 编辑、API 覆盖写、CLI 覆盖写和 agent append 入口，但覆盖保存没有声明“基于哪个
`updated_at` 保存”。如果用户打开 Run Inspector 后，agent 通过 `append-run-note` 追加复现线索，用户再点击旧页面里的保存，
旧草稿会覆盖 agent 刚写入的内容。

### 根因

`note.json` 已经是独立 artifact 真源，但 `update_run_note()` 只做长度校验和原子写入，没有可选的乐观并发 guard。
前端保存和 CLI `set-run-note` 也没有把读取到的 `updated_at` 带回后端。

### 影响范围

- 影响人类和 agent 同时维护同一份 run note 时的复现线索稳定性。
- 不改变 eval、codec、metric、rank-board、job 或数据语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- `EvalBenchStore.update_run_note()` 新增可选 `expected_updated_at`，不匹配当前 note 时抛出 `RunNoteConflictError`。
- Dashboard `PATCH /api/runs/{run_id}/note` 支持 `{ "expected_updated_at": ... }`，冲突时返回 HTTP 409。
- CLI `set-run-note` 新增 `--expected-updated-at`，agent 可先 `get-run-note` 再做 guarded 覆盖；`append-run-note`
  仍作为增量记录入口。
- Run Inspector 保存备注时带上当前 `run.note_updated_at`，避免旧页面静默覆盖新备注。
- 前端 API client 新增带 `status` 的 `ApiError`；Run Inspector 遇到 409 冲突时用结构化状态判断并主动刷新
  `dashboard-state`，把最新 note 拉回页面，不把旧草稿继续当作当前真源。
- README、架构文档、Dashboard 测试、CLI 测试和 UI contract 同步该并发边界。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_updates_editable_run_note projects/eval_bench/tests/test_cli.py::test_cli_parser_commands_have_handlers_for_agent_contract projects/eval_bench/tests/test_cli.py::test_cli_gets_and_sets_run_note`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`

### 后续防线

- 覆盖式 run artifact 写入必须考虑 expected version；agent 增量写入优先提供 append/patch 语义，避免读改写覆盖。
- 前端保存失败遇到 409 时必须基于 `ApiError.status` 刷新 state，不能靠错误文案字符串判断，也不能吞掉冲突。

## 2026-05-25: Eval Bench 高级检索展开后仍挤压主工作区

### 现象

`AdvancedFilterBar` 已经把多组 select 收进 Filter 入口，但展开后的检索表单仍在文档流里占高度。
Runs、Jobs、Rank Board 和检查器侧栏在展开高级检索后，主表格或样本区域会被向下挤，用户还需要再滚动才能回到核心内容。
生效条件只能统一清空，不能像论文检索条件那样直接删除某一个条件。

### 根因

共享高级检索组件只完成了“折叠入口”和“统一清空”的第一层收敛，没有把展开态做成真正的工作区浮层。
条件摘要也只是静态 chip，单条件清除逻辑没有收敛到同一个默认值判定函数。

### 影响范围

- 影响所有复用 `AdvancedFilterBar` 的目录页、排行榜和 inspector 样本过滤交互。
- 不改变 eval、codec、metric、data 或查询 API 语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- `AdvancedFilterBar` 的展开表单改为相对当前 filter 条的绝对定位浮层，设置最大高度和内部滚动，避免撑高主工作区。
- 生效条件 token 改为可点击的 `ActionButton`，点击后只清除对应条件，并复用同一个 `defaultFilterValue` / `resetAdvancedFilter` 真源。
- Inspector 侧栏保留静态展开流和局部滚动，避免窄侧栏浮层遮挡样本列表。
- README、架构文档、脚本文档和 layout smoke 同步新的 token 清除与浮层滚动边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- 新增页面级 filter 必须复用 `AdvancedFilterBar`，不能把高级检索表单直接插进页面主流导致核心内容位移。
- 新增 filter 默认值或条件类型时，必须同时接入单条件 token 清除和统一清空逻辑。

## 2026-05-25: Eval Bench 首页 command desk 仍缺少主判断链路

### 现象

总览页已经多次删除低价值面板，但用户仍反馈“主页没有任何价值”。当前首屏把 hero、四个信号、管线、
阻塞优先级、行动入口和最近 run 分散摆放，页面虽然不再是图表墙，但视觉上仍像多个模块并列。
鼠标悬浮和同步状态动效也不够统一，系统缺少实时控制台的互动感。

### 根因

上一版把“卡在哪里”拆成独立阻塞优先级面板，又把信号、管线和行动入口分成多组，导致用户需要自己把这些块重新合成下一步判断。
布局契约也仍以旧 command desk 为中心，约束了元素存在，但没有约束首页必须先给出主判断和主动作。

### 影响范围

- 影响 Eval Bench Dashboard 首页第一屏判断效率、空间层级和交互质感。
- 不改变 eval、codec、metric、data、store 或 rank-board 语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- 总览升级为 `overview-home-v7`，改成 home cockpit：左侧 priority stage 直接给出当前判断、同步状态、
  下一步主动作和四段路线图。
- 右侧 command rail 只保留覆盖、待评、队列和服务四个可点击状态入口，下方保留一个评测流向 surface、
  readiness switchboard 和最近 run 紧凑摘要。
- 移除单独常驻的阻塞优先级面板，卡点由主动作和状态入口表达，避免首页继续并列堆模块。
- 补充统一 hover、live pulse、sync sheen、flow sweep 和卡片抬升动效，动效只表达可点击性和实时状态。
- 同步 README、架构文档、脚本说明、静态 UI contract 和 layout smoke 的新首页边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- 首页新增内容必须能直接强化“当前判断、下一步动作、评测流向、最近产物”之一；不能新增独立低价值面板。
- Overview 契约必须同时约束源码结构、运行时高度和 hover/transition，不只检查 selector 存在。

## 2026-05-25: Eval Bench detection label 子任务仍能自由输入 label

### 现象

后端已经通过 `label_policy.py`、preflight、worker、import 和 evaluator 收敛了 detection label subset 与
keypoint arrow-only 语义，文档也要求导入 prediction run 不暴露自由文本 label 输入。但共享
`DetectionLabelSubtaskPanel` 仍保留“自定义 label”输入，导致 Jobs 页和 Runs 导入弹层都能绕过候选 chip
拼写任意 label。

### 根因

前期为了兼容 benchmark 没有 label index 的情况，在共享面板里留下了自由文本添加入口。后续 label policy
和 preflight 已经成为真源，但 UI 没有同步收口，形成了与文档和稳定性目标不一致的第二入口。

### 影响范围

- 影响 Dashboard detection label 子任务创建和 prediction import 的输入稳定性。
- 不改变 eval、codec、metric 或后端 label policy 语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- 移除 `DetectionLabelSubtaskPanel` 的自由文本输入和添加表单，只保留候选 chip、全部候选和默认策略。
- 候选仍来自 benchmark summary、prompt metadata 和当前 manifest，未知 label 必须通过 manifest/payload
  显式编辑并交给 preflight 或 import 校验。
- 删除对应 dead CSS，UI contract 增加防线，禁止 label 子任务面板回流自由文本 label 入口。
- README、架构文档和脚本文档同步该 UI 边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run test:manifest-tools`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- 新增 label 子任务 UI 时只能复用受控候选 chip，不要在页面层新增自由文本 label 输入。
- 如果 benchmark 缺少 label index，需要先补 benchmark metadata 或让用户显式编辑 manifest，再由后端 preflight 校验。

## 2026-05-25: Eval Bench Rank Board 缺少与榜首的差距解释

### 现象

Rank Board 已经支持默认 F1 主指标、主指标切换、显式 weighted scheme、facet 和后端分页，但 entry 只返回
当前分数。用户或 agent 看到第 2 名、第 10 名时，需要自行拿榜首分数做差，分页后更容易只拿当前页第一条误判差距。

### 根因

上一版把排序、分页和 weighted score 的真源放在 store/API/CLI，但没有把“相对当前榜首的主分数差值”作为
rank-board 输出契约的一部分。前端 score spread 只能描述当前页分布，不能解释每条 run 和全局当前第一名的距离。

### 影响范围

- 影响 Rank Board 核心页面、CLI/API 输出和 agent 对排行结果的可解释性。
- 不改变 eval、codec、metric、data 或 weighted score 计算语义；这不是模型能力问题，也不是评估标准误判。

### 修复方式

- `RankBoardEntry` 新增 `score_delta`，由 store 在完整排序后按 entry `score - leader.score` 计算。
- 分页前完成 delta 计算，因此 offset 页仍以完整当前排行第一名为基准，而不是当前页第一条。
- CLI `rank-board` 和 `/api/rank-board` 通过同一 dataclass 序列化输出 `score_delta`。
- Rank Board 表格新增 `Delta leader` 列，Top contenders 也显示每条 run 的 leader-relative delta。
- README、架构文档、脚本文档、dashboard/CLI 测试和 UI contract 同步该输出契约。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_exposes_independent_rank_board projects/eval_bench/tests/test_cli.py::test_cli_prints_filtered_rank_board`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- Rank Board 新增排序或主指标时，必须同步检查 `entry.score`、`primary_metric_label` 和 `score_delta` 是否仍基于同一主分数。
- 前端不能用当前页局部第一条自行计算 leader gap；差距解释必须来自 store/API/CLI 真源。

## 2026-05-25: Eval Bench Agent 命令只能全量发现不能单条查询

### 现象

Agent 可以通过 `list-agent-commands` 获取完整稳定命令面，但如果只想执行某个命令，例如 `rank-board`
或 `append-run-note`，仍需要拉取全量列表后自行过滤。随着 CLI 命令数量增加，这会增加上下文噪声，
也让 agent 更容易从自然语言 help 或旧缓存里拼 argv。

### 根因

前期只实现了“全量枚举稳定命令”的发现入口，没有把单条命令契约抽成可复用 helper 和 CLI 子命令。
`list-agent-commands` 内部直接拼装契约，导致没有一个可被单命令查询复用的真源。

### 影响范围

- 影响 agent 在低上下文预算下查询单个命令参数、互斥组和副作用标记。
- 不影响 Dashboard 前端、eval/codec/metric/data 语义或已有 CLI 命令执行结果。
- 这不是模型能力问题，也不是评估误判；属于 agent CLI 契约可发现性问题。

### 修复方式

- 新增稳定 CLI：`show-agent-command --name <command>`，输出单条命令的 `domain`、`mutates_state`、
  `destructive`、`usage`、`argv_prefix`、`arguments` 和 `mutually_exclusive_groups`。
- 将 agent 命令契约生成收敛到 `_agent_command_contracts()` / `_agent_command_contract()`，
  `list-agent-commands` 和 `show-agent-command` 复用同一套 argparse-derived schema。
- 将 `show-agent-command` 登记进 `AGENT_COMMAND_METADATA` 和 `_command_handlers()`。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_cli.py`

### 后续防线

- 新增稳定 agent 命令后，必须同时能被 `list-agent-commands` 枚举，并能被 `show-agent-command --name`
  单条查询；不要让 agent 从自然语言 help 或前端状态猜参数。

## 2026-05-25: Eval Bench 弹窗只统一了外壳但没有统一焦点和滚动语义

### 现象

Dashboard 业务页已经统一使用 `WorkspaceDialog` 外壳，但弹窗打开后仍缺少明确的焦点进入、Tab 焦点闭环、
body scroll lock 和关闭后的焦点恢复验收。后续业务页如果为了修复单个弹窗体验各自补 effect，
会重新出现弹窗交互语义分叉。

### 根因

前期弹窗收口只把 class、标题、关闭按钮、Escape/backdrop 和 body 滚动容器合并到 `WorkspaceDialog`，
没有把可访问性和页面滚动锁作为共享原语的一部分；dialog smoke 也只检查能打开和 Escape 关闭，
没有验证焦点与页面滚动状态。

### 影响范围

- 影响 Jobs、Benchmarks、Runs、Services 和危险确认弹窗的交互一致性。
- 影响窄屏或长表单弹窗打开时页面滚动和键盘焦点稳定性。
- 不影响 eval、codec、metric、data 或 store 语义；这不是模型能力问题，也不是评估误判。

### 修复方式

- `WorkspaceDialog` 增加 body scroll lock、初始焦点、Tab 焦点闭环、关闭后焦点恢复和 `aria-describedby`。
- `dialog-smoke-check` 打开每类弹窗后检查焦点在弹窗内、body overflow 被锁住、Tab 不逃逸，关闭后恢复滚动。
- `layout-smoke-check` 在弹窗布局验收中同步检查焦点与 body scroll lock。
- `test:ui-contracts` 静态锁住 `WorkspaceDialog` 的焦点、滚动锁和可访问性 wiring。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:dialogs`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- 新增弹窗只允许组合 `WorkspaceDialog`，不能在业务页直接写 body overflow、keydown focus trap 或 dialog shell class。
- 如果需要 footer、危险确认或 wizard 行为，应扩展共享弹窗原语或共享子组件，而不是在单个业务页复制交互 effect。

## 2026-05-25: Eval Bench 首页仍缺少可行动价值和整体交互感

### 现象

总览页已经收敛到 command desk，但首屏仍像多个状态模块的组合：hero、四个信号、pipeline、
阻塞优先级、行动入口和最近 run 之间缺少明确的主判断链路。用户进入首页后仍觉得没有价值，
同时 dashboard 级 hover、入场和在线状态动效不够统一。

### 根因

上一版只把低价值面板删掉，没有把首页重新设计成“下一步决策优先”的界面。pipeline 和阻塞信息虽然存在，
但视觉上仍像两个面板；最近 run 仍偏状态列表；通用动效散落在局部组件，没有形成系统级交互反馈。

### 影响范围

- 影响 Eval Bench Dashboard 首页第一屏判断效率和视觉层级。
- 影响全局按钮、导航、卡片、状态胶囊和表格的交互一致性。
- 不改变 eval、codec、metric、data 或 store 语义；这不是模型能力问题，也不是指标误判。

### 修复方式

- 总览升级为 `overview-home-v6`：顶部 hero 增加 benchmark -> run -> report 路线提示和技术路线图，
  让当前系统态、数量和下一步动作在同一屏建立主线。
- operations surface 改为“可行动信号 + pipeline / bottleneck flow”结构，管线和卡点放在同一个 flow surface，
  不再像互不相关的面板墙。
- 最近 run 改为紧凑产物摘要，只展示 benchmark/model 与 prediction/report 数量，减少无价值状态噪声。
- `design.css` 增加共享入场、hover、图标抬升和 live pulse 动效，覆盖标准 workspace 元素。
- UI contract 锁住 `overview-home-v6`、`OverviewHeroMap`、`overview-flow-and-bottleneck` 和共享动效关键帧。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ SCREENSHOT_PATH=/home/tanjingyuan/code/arrow-vlm/temp/eval_bench_overview_v6.png npm run render-check`

### 后续防线

- 首页新增内容必须能直接回答“下一步去哪”或“当前卡在哪里”；不能把低频诊断或字段摘要搬回总览。
- 动效只服务可点击性、实时状态和局部层级反馈，不能制造布局抖动或替代信息结构。

## 2026-05-25: Eval Bench 导入预测仍使用自由文本 label 子任务输入

### 现象

新建 eval job 已经有 detection label 子任务 chip，keypoint 不暴露任意 label 子集。
但 Runs 页的导入 prediction run 仍使用“目标标签”自由文本输入。人类容易手输拼错 label，
keypoint 路径也会看到和 detection 子任务类似的输入，前端体验与后端 label policy 不一致。

### 根因

label 子任务面板最初实现为 Jobs 页局部组件，没有抽成共享控制原语。导入 prediction 弹窗为了快速可用，
保留了字符串解析 `parseTargetLabels()`，把 label 子任务 UI 和 import payload 语义分开维护。

### 影响范围

- 影响 prediction import 的 detection label 子任务选择体验。
- 影响 keypoint import 的 UI 语义清晰度，但后端仍会拒绝非法 keypoint label。
- 不影响 evaluator、worker、rank-board 或已有 run note。
- 这不是模型能力问题；属于 eval UI 与 label policy 的语义一致性问题。

### 修复方式

- 新增共享 `DetectionLabelSubtaskPanel`，集中维护候选 chip、全部候选、默认策略和自定义 label。
- Jobs 页和 Runs 导入 prediction 弹窗共同复用该组件。
- Runs 导入 prediction 不再解析自由文本目标标签，直接提交 `target_labels: string[]`。
- task 切到 keypoint 时自动清空并隐藏 label 子任务；切换 benchmark 时按 benchmark label index 收敛已选 label。
- UI contract 覆盖共享组件和 Runs import 对自由文本 target label 输入的禁止回流。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_prediction_import.py projects/eval_bench/tests/test_cli.py`

### 后续防线

- 新增 label 子任务入口必须复用 `DetectionLabelSubtaskPanel` 或同层共享组件，不在业务页手写自由文本 label 输入。
- keypoint UI 不展示任意 label 子任务入口；后端继续在所有入口拒绝非 `arrow` 显式 label。

## 2026-05-25: Eval Bench Agent 追加 run note 只能覆盖整份备注

### 现象

Eval Bench 已有 `get-run-note` 和 `set-run-note`，Dashboard 也能编辑同一份 run note。
但 agent 如果只想追加一条复现线索、idea 来源或排障记录，只能先读出整份 note、在外部拼接、再用
`set-run-note` 覆盖。这个流程容易覆盖人类刚写入的内容，也鼓励 agent 直接改 `note.json`。
前端编辑器也只有自由文本框，缺少面向复现记录的结构化模板入口。

### 根因

run note 真源已经收敛到 store，但 store/CLI/API 只暴露了覆盖写，没有暴露 append 语义。
UI 也只实现了文本编辑，没有把常见的复现、idea、异常和 next 记录模式固化为组件。

### 影响范围

- 影响 agent 持续记录实验线索的安全性和可维护性。
- 影响人类在 Run Inspector 中快速写出结构化 note。
- 不影响 eval metric、rank-board 排序、prediction snapshot 或 run manifest。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- 在 store 中新增 `append_run_note()`，复用同一份 `note.json` 真源和长度校验。
- 新增 agent-safe CLI `append-run-note`，支持 `--note` / `--note-file` 和可选 `--heading`，
  追加带 heading 的 markdown 段落而不是覆盖整份 note。
- Dashboard API 新增 `POST /api/runs/{run_id}/note/append`，HTTP agent 也可以复用同一份 append 语义。
- Run Inspector 的 note 编辑器新增复现、Idea、异常、Next 模板插入按钮，统一使用 `ActionButton`。
- UI contract 和 layout smoke 覆盖模板入口，CLI/Dashboard 测试覆盖 append 后再读取的完整 note。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_cli.py projects/eval_bench/tests/test_dashboard.py`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- agent 追加实验线索优先用 `append-run-note` 或 `/note/append`；只有人工整理最终版本时才用 `set-run-note`
  或 `PATCH /note` 覆盖。
- 新增 note 字段或 UI 模板时必须继续经过 store/API/CLI 真源，不直接读写 artifact 文件。

## 2026-05-25: Eval Bench Rank Board 主指标控制藏在高级检索里

### 现象

Rank Board 已经从 Compare 中独立出来，并支持 F1 默认排序、切换主指标和 weighted scheme。
但主指标切换与升降序控制仍主要位于高级检索表单里，核心排行榜首屏缺少直接表达“当前排名依据”的工作区。
用户需要展开 Filter 才能改排名依据，页面更像结果表而不是核心 rank-board。

### 根因

前期优先把排序语义接到后端 `fetchRankBoard()` 和分页表格，没有把“排名依据”提升为 Rank Board 自身的首层交互。
高级检索同时承载筛选条件和主指标控制，导致核心操作和检索条件混在一个浮层里。

### 影响范围

- 影响 Rank Board 的核心使用体验和可解释性。
- 不影响后端排序、weighted scheme 计算、CLI/API、run note 或 eval metric。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- 新增 `RankDecisionPanel`，首层展示当前 ranking basis、主指标 chip、升降序、score formula。
- 新增 Top contenders 和当前页 score spread，让 rank-board 不只是表格。
- 将 `rank-sort-by` / `rank-sort-order` 从高级检索中移出；高级检索只保留筛选和最低分门槛。
- 更新 layout smoke 和 UI contract，防止主指标控制回流高级检索浮层。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- Rank Board 的排名依据必须保持在页面首层；高级检索只负责检索范围。
- 新增主指标或排序方式时，需要同时更新 `RankDecisionPanel`、后端 rank-board API/CLI 和 UI contract。

## 2026-05-25: Eval Bench 高级筛选仍占用页面主内容空间

### 现象

Runs、Compare、Rank Board、Jobs、Services、Benchmarks 和样本检查器已经复用 `AdvancedFilterBar`，
但旧形态展开时仍在页面主内容区铺开大量 select/input。筛选条件越多，越容易挤压排行榜、结果表和样本列表，
与“论文检索式高级检索”以及 Filter 按钮收敛的目标不一致。

### 根因

高级筛选的真源只收敛了控件和清空逻辑，没有把布局语义也收敛为“触发器 + 摘要 + 分组浮层”。
业务页虽然不再各自堆 raw select，但共享组件本身仍保留占位式面板布局。

### 影响范围

- 影响 dashboard 前端的信息密度、滚动稳定性和筛选交互一致性。
- 不影响后端分页、CLI/API 查询语义、rank-board 指标计算、run note 或 eval metric。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- `AdvancedFilterBar` 默认压缩为 Filter 触发器、当前条件摘要和统一清空动作。
- 展开态改为浮层式高级检索表单，并按“检索式 / 范围目录 / 排序与阈值”自动分组。
- 新增 Escape 和外部点击关闭行为，保留所有页面已有筛选状态和后端查询参数。
- README、架构文档和静态 UI contract 同步新的筛选组件边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:layout`

### 后续防线

- 新页面新增筛选必须复用 `AdvancedFilterBar`，不能重新铺开一排业务私有 select/input。
- `test:ui-contracts` 必须继续约束高级筛选的浮层、分组目录和统一清空动作，避免回退到占位式筛选面板。

## 2026-05-25: Eval Bench preflight 未展示最终 label scope 且 legacy prompt 匹配过宽

### 现象

agent 使用 `preflight-job` 时，`resolved_payload` 不一定展示运行时最终会使用的 `target_labels`。
例如 keypoint job 缺省 target labels 时，worker 会按任务补成 `arrow`，但 preflight 输出可能还是空。
同时 legacy prompt ID 规则用任意 `layout` / `arrow` 子串推断 label scope，导致自定义 `custom.layout`
也被误判成 layout prompt，并在 preflight 中触发不存在的 `image/shape` label 错误。

### 根因

`job_spec.resolve_job_payload()` 只做 manifest/runtime 字段归一化，没有复用运行时 `label_policy`。
而 `label_policy` 的 legacy prompt 兼容规则过宽，把普通自定义 prompt ID 当成内置 prompt 家族。

### 影响范围

- 影响 agent 通过 `preflight-job` 判断 detection label 子任务和 keypoint arrow-only 作用域。
- 影响自定义 prompt ID 中包含 `layout`、`arrow` 等普通词时的 label scope 推断。
- 不影响模型能力；这是 eval/job preflight 与 label policy 的语义一致性问题。

### 修复方式

- `resolve_job_payload()` 在返回 `resolved_payload` 前应用同一套 `resolve_target_label_policy()`，
  补齐 `target_labels` 和 `target_labels_source`。
- keypoint 缺省 target labels 会在 preflight 阶段显示 `arrow`，来源为 `task_default` 或内置 prompt 的
  `legacy_prompt_id`。
- legacy prompt ID 收窄到内置命名族：`grounding_layout.*`、`grounding_arrow.*`、`keypoint_arrow.*`
  和历史 `arrow_keypoint.*`，不再匹配任意 `custom.layout`。
- README 和架构文档同步 agent preflight 与 legacy prompt ID 边界。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_eval_semantics.py projects/eval_bench/tests/test_cli.py`
- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py projects/eval_bench/tests/test_prediction_import.py projects/eval_bench/tests/test_evaluator.py`

### 后续防线

- 新增 prompt ID 兼容规则时必须限定在已知内置命名族；自定义 prompt 的 label scope 只能来自 metadata 或显式 run spec。
- agent-facing preflight 输出必须和 worker/import/evaluator 最终语义一致，不能把运行时默认值藏到入队后。

## 2026-05-25: Eval Bench Overview 静态 UI 契约仍锁旧结构

### 现象

总览页已经从旧的 `overview-command-deck` / `overview-focus-panel` / 活动矩阵改成 command desk，
但 `test-ui-contracts.mjs` 仍要求旧 class 和 `OverviewActivityMatrix` 存在。这样会让后续静态检查把旧结构误认为正确形态。

### 根因

上一轮只更新了运行时 layout smoke，没有同步静态 UI contract。Overview 的结构边界同时由源码模块和 smoke 检查约束，
只更新其中一个会形成双轨标准。

### 影响范围

- 影响 Overview 模块的长期重构防线和后续 UI 组件化约束。
- 不影响 dashboard API、run/report/rank 数据或评测指标。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- `test:ui-contracts` 改为要求 `overview-command-center-redesign`、`overview-workbench`、`overview-ops-surface`、
  `overview-bottleneck-panel` 和 `overview-right-rail`。
- 静态契约明确禁止 Overview 源码回流 `overview-command-deck`、`overview-focus-panel`、`overview-side-stack`、
  `OverviewActivityMatrix` 和 `overview-activity-matrix`。
- README 与脚本文档同步这条静态防线。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`

### 后续防线

- Overview 布局重构必须同时更新 runtime layout smoke 和 static UI contract；不能只改其中一层。

## 2026-05-25: Eval Bench Overview 活动矩阵和面板墙降低首页价值

### 现象

总览页经过多轮压缩后仍保留 Run/Job/Service 活动矩阵、多个卡片版本样式和 6 条最近 run 摘要。
这些模块占用首屏高度，却不能直接回答当前是否可用、卡在哪里、下一步去哪；在 compact / narrow 视口下还容易把核心面板压成不可读外壳。

### 根因

Overview CSS 和布局契约沿用早期“多图表总览”的思路，把写入节奏和状态分布当成首页核心价值。
实际运营场景更需要优先级判断和可跳转入口；旧 layout smoke 还要求活动矩阵存在，导致低价值模块被测试固化。

### 影响范围

- 影响 dashboard 总览页的信息密度、首屏可读性和鼠标交互反馈。
- 不影响后端 state/job/service/scheduler API、Rank Board、run report 或评测指标。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- 总览改成 command desk：顶部 hero 展示当前系统态，右侧 dock 展示同步态和当前优先动作。
- 左侧 operations surface 保留四个可行动信号、四段管线和新增阻塞优先级条；删除旧活动矩阵。
- 右侧只保留 readiness switchboard 和最近 4 条 run 紧凑摘要。
- 增强首页与标准 workspace 元素的 hover、pulse、rail transition 和入场动画，动效只表达可点击性和实时状态。
- layout smoke 改为检查阻塞优先级、operations surface、可读高度和旧活动矩阵不回流。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ SCREENSHOT_PATH=/tmp/eval-overview.png npm run render-check`

### 后续防线

- Overview 新增模块必须服务“当前是否可用、卡在哪里、下一步去哪”，不能把低频诊断、活动日历或细粒度 metric 放回首页。
- compact / narrow 视口下必须由 Overview 页面栈承担滚动，核心面板不能再次被 flex shrink 压成 30-40px 外壳。

## 2026-05-25: Eval Bench agent 命令缺少危险写入标记

### 现象

`list-agent-commands` 已经输出 `domain`、`mutates_state` 和参数 schema，但 agent 只能知道某个命令会写状态，
不能区分普通写入和删除、归档、取消、停止这类危险生命周期操作。比如 `create-job`、`set-run-note`
和 `delete-run` 都只是 `mutates_state=true`，agent 仍需要从命令名猜风险。

### 根因

早期 agent metadata 只覆盖只读/写入二元分类，满足了“不要 hack CLI”的基础发现面，但没有把危险副作用建模为
一等元数据。生命周期命令扩展后，单一 `mutates_state` 已经不够表达操作风险。

### 影响范围

- 影响 agent 对 run/job/service/prompt 生命周期命令的风险判断。
- 不影响命令执行逻辑、API、dashboard 或 artifact 格式。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- 新增 `AGENT_DESTRUCTIVE_COMMANDS`，标记 `archive-run`、`cancel-job`、`delete-job`、
  `delete-prompt-template`、`delete-run`、`delete-service` 和 `stop-service`。
- `list-agent-commands` 每条命令输出 `destructive`，顶层输出 `destructive_count`。
- CLI contract 测试确认 destructive 集合属于稳定命令面，且 destructive 命令必须同时是 mutating。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_cli.py::test_cli_parser_commands_have_handlers_for_agent_contract projects/eval_bench/tests/test_cli.py::test_cli_lists_agent_stable_commands`

### 后续防线

- 新增会删除、归档、取消、停止或清理资源的 agent 命令时，必须同时进入 `AGENT_DESTRUCTIVE_COMMANDS`；
  普通写入只设置 `mutates_state`，不要让 agent 从自然语言 help 里推断风险等级。

## 2026-05-25: Eval Bench AdvancedFilterBar 折叠头仍使用 raw button

### 现象

大部分业务动作已经收敛到 `ActionButton`、`IconActionButton`、`OptionChipButton` 和
`SelectableCardButton`，但共享 `AdvancedFilterBar` 的折叠头仍直接写原生 `<button>`。
它是所有高级检索入口的通用交互，如果继续保留 raw button，后续 hover、active、aria 和样式层级容易继续漂移。

### 根因

早期收口只覆盖了业务页里的按钮和高级检索的清空动作，没有给“折叠面板入口”建一个独立按钮原语。
`AdvancedFilterBar` 虽然是共享组件，但它内部仍绕过 `ui.tsx` 的按钮体系。

### 影响范围

- 影响 Runs、Benchmarks、Jobs、Services、Compare、Rank Board 和 inspector 筛选入口的交互一致性。
- 不影响筛选查询、API、分页、排序或评测指标。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- `ui.tsx` 新增 `PanelToggleButton`，统一折叠/展开按钮的 `aria-expanded`、active class 和基础 button 语义。
- `AdvancedFilterBar` 的 `.advanced-filter-head` 改用 `PanelToggleButton`，保留现有视觉样式和展开逻辑。
- UI contract 增加静态防线：`filterControls.tsx` 必须导入并使用 `PanelToggleButton`，不能再出现
  `advanced-filter-head` raw button。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 新增折叠面板、抽屉入口或高级筛选头时，先复用 `PanelToggleButton`；业务页不直接写 raw `<button>`
  来模拟可展开标题。

## 2026-05-25: Eval Bench Jobs 应用 Prompt 后 benchmark 未按 task 重选

### 现象

Jobs 页从 detection manifest 切换到 keypoint prompt 时，manifest 的 `eval.task` 会被更新为
`keypoint`，但原来的 `benchmark_id` 只要仍存在于 benchmark 列表中就会被保留。若该 benchmark
只支持 detection，用户会得到一个 task / benchmark 不匹配的 job draft，需要依赖后端 preflight
再报错。

### 根因

前端 `applyBenchmarkDefault()` 只检查当前 benchmark id 是否存在，没有检查该 benchmark 的
`tasks` 是否包含当前 manifest task。`applySelectedPrompt()` 应用 prompt 后也没有重新跑一次
task-aware benchmark default，导致 prompt 改 task 后 benchmark 不随之纠正。

### 影响范围

- 影响 Jobs 页 manifest-first 创建任务时 detection / keypoint prompt 切换的默认 benchmark 选择。
- 不影响后端 `label_policy.py`、preflight 的非法 label 拒绝、worker 或 evaluator 语义。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- `applyBenchmarkDefault()` 改为 task-aware：当前 benchmark 不存在或不支持当前 task 时，选择第一个支持该 task 的 benchmark。
- Jobs 页应用 prompt 后重新调用 `applyBenchmarkDefault()`，让 prompt task、target labels 和 benchmark default 同步收敛。
- manifest tools 回归测试覆盖“当前 benchmark 存在但 task 不匹配”的重选场景。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:manifest-tools`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 任何会修改 manifest task 的前端操作，都必须同步检查 benchmark task 兼容性，不能只检查 benchmark id 是否存在。

## 2026-05-25: Eval Bench Overview compact 视口核心面板塌缩

### 现象

Overview 首页在 desktop 下还能看到总控信息，但在 compact / narrow 视口下，
`overview-focus-panel` 被压缩到 35-40px 高，评测闭环、四个信号卡、pipeline 和活动矩阵都进入不可见区域。
用户看到的是一组被压扁的 block，无法作为总控工作台使用。

### 根因

Overview 经过多轮改版后，CSS 中保留了 v1/v2/v3 多套空间规则。`max-width: 1180px`
媒体查询把 command deck 改成 column 并让它参与 flex 剩余空间分配，后续规则又没有明确覆盖
`flex` 和 overflow 语义，导致核心面板在固定高度 `content` 容器中被压缩。layout smoke 只检查元素存在和高度大于 0，
没有检查核心面板是否达到可读高度。

### 影响范围

- 影响 Overview 首页在 compact / narrow 视口下的可读性和操作价值。
- 不影响后端 state/job/service/scheduler API、Rank Board、run report 或评测指标。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- 增加 Overview v4 空间规则：desktop 使用两列 command surface；compact 下 hero 与核心面板完整展开，
  readiness 和最近 run 横向平分；narrow 下按单列堆叠，由 `.dashboard-home` 承担滚动。
- 明确 command deck 使用 `flex: 0 0 auto` 和 `overflow: visible`，避免核心模块被内部滚动容器裁切。
- 顶部 pulse dock 改成调度、失败、运行、在线四个运行态信号，减少和报告覆盖信号的重复。
- 修正“待评估为 0 但仍显示已有预测”的误导性文案，在线服务文案改为服务登记状态。
- layout smoke 新增 Overview 面板可读高度断言，防止 focus/action/recent 面板再次塌缩。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- Playwright 截图检查 desktop / compact / narrow 三种视口，确认核心面板高度不再塌缩。

### 后续防线

- Overview 布局变更不能只检查 selector 存在；必须检查核心控制面板的可读高度、滚动归属和 compact/narrow
  下的空间算法。后续应继续清理旧 Overview CSS 规则，避免多套版本规则叠加。

## 2026-05-25: Eval Bench Rank Board facet rail 漏接任务与状态筛选

### 现象

Rank Board 后端 `/api/rank-board` 已经返回 Tasks、Benchmarks、Status、Labels、Models、Prompts
和 Metrics 七类 facet，但前端 facet rail 只把 Labels、Models、Prompts 和 Metrics 接成可点击 chip。
用户要按任务、基准集或 run 状态收窄排行榜时，仍必须回到高级检索面板手动选择。

### 根因

上一轮把 facet rail 从静态计数改成可点击交互时，只覆盖了原来已经显示在 rail 里的四类 facet，
没有以 API facet 真源为边界做完整对齐。高级检索状态本身已经有 task、benchmark 和 status，
前端遗漏的是同一状态的第二入口 wiring。

### 影响范围

- 影响 Rank Board 核心页面的检索效率和交互一致性。
- 不影响后端 facet 计算、排行榜排序、weighted scheme、CLI/API 输出或 report 指标。
- 这不是模型能力问题，也不是 eval/codec/metric/data 误判。

### 修复方式

- Rank Board facet rail 现在暴露 Tasks、Benchmarks、Status、Labels、Models、Prompts 和 Metrics。
- 新增的三个 facet chip 直接更新同一份高级检索状态；再次点击 active chip 回到 `all`。
- facet rail 保持最多四列，较窄视口降到两列/一列，并补充 hover/active 动效。
- UI contract 和 layout smoke 锁住七类后端 facet 都必须作为可点击 `.rank-facet-button` 出现。
- README、架构文档和脚本文档同步 facet rail 的完整交互边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增 Rank Board facet 时，必须以 API 返回的 facet group 为真源，先接入同一份高级检索状态，再更新
  UI contract 和 layout smoke；不能只补一个摘要面板。

## 2026-05-25: Eval Bench Rank Board 主指标切换后表格仍固定显示 F1

### 现象

Rank Board 后端已经支持 `sort_by=f1_iou50|precision_iou50|recall_iou50|mean_iou` 和 weighted scheme，
并且返回 `primary_metric_label` 与每条 entry 的当前 `score`。但前端表格的第一分数字段仍固定是
`F1@.50`；用户切换主指标后，排序依据已经改变，表格视觉上却仍像按 F1 排名。

### 根因

前端最初把 F1 作为默认主指标写死在表格列里，后续补主指标切换和 weighted scheme 时只更新了 API、
排序和顶部 formula chip，没有把表格第一分数列绑定到后端的 `primary_metric_label` / `score` 真源。

### 影响范围

- 影响 Rank Board 核心页面对当前排名依据的可解释性。
- 不影响后端排序、CLI `rank-board`、weighted score 计算或 report 指标。
- 这不是模型能力问题，也不是 eval/codec/metric 误判。

### 修复方式

- Rank Board 表格第一分数列改为动态主指标列，列名来自 `primary_metric_label`，值来自 entry `score`。
- 当主指标不是 F1 时，额外保留 `F1@.50` 作为诊断列；默认 F1 模式不重复显示两列。
- Weighted scheme 模式复用同一主指标列，并继续展示 score components。
- UI contract 锁住 `rank-primary-score` 和动态主指标列，禁止回退到固定 `Weighted` / 固定 F1 主列。
- README、架构文档和脚本文档同步 Rank Board 主指标展示契约。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- Rank Board 新增排序字段或主指标时，必须确保 API 的 `score`、`primary_metric_label` 和表格第一分数列
  同步变化；固定指标列只能作为诊断列存在。

## 2026-05-25: Eval Bench Rank Board facet rail 只是静态计数

### 现象

Rank Board 已经独立成核心页面，并且有高级检索条，但页面下方 facet rail 只展示 Labels、Models、
Prompts 和 Metrics 的前几个计数。用户看到 facet 后仍要回到折叠检索表单手动选择同一条件，不符合
“论文检索式”排行榜工作台的交互预期。

### 根因

facet rail 最初作为 rank-board 的轻量摘要加入，没有把它接回高级检索状态。`AdvancedFilterBar`
是真正检索入口，facet rail 则变成第二套只读展示，造成 UI 语义重复。

### 影响范围

- 影响 Rank Board 的检索效率和核心页面设计完整性。
- 不影响后端 `/api/rank-board`、CLI `rank-board`、排序、weighted scheme 或 run report 指标。
- 这不是模型能力问题，也不是 eval/codec/metric 误判。

### 修复方式

- Rank Board facet rail 改用 `OptionChipButton` 渲染可点击 facet chip。
- Labels、Models、Prompts 和 Metrics facet 点击后直接更新同一份高级检索状态；再次点击 active chip
  回到 `all`。
- UI contract 锁住 facet rail 必须使用可点击 chip，而不是静态 count。
- layout smoke 在 Rank Board 路由点击 facet chip，确认 active 状态和 table 仍然可用。
- README、架构文档和脚本文档同步 facet rail 的交互边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- Rank Board 新增 facet 时必须先接入同一份高级检索状态；不能新增只展示、不驱动查询的 facet 面板。

## 2026-05-25: Eval Bench 稳定 agent 命令面漏掉已有关键 CLI

### 现象

`init-run`、`validate-prediction` 和 `process-next-job` 已经是可用 CLI，但没有进入
`list-agent-commands` 的稳定命令面。Agent 通过正式发现入口只能看到 create/import/evaluate/list 等
命令，无法稳定发现“初始化 run manifest”“校验单个 prediction 文档”和“手动推进下一个 queued job”。

### 根因

这些命令早于 agent command metadata 收口存在，后续补 `AGENT_COMMAND_METADATA` 时只覆盖了主要目录、
service、rank、comparison 和 note 入口，没有把已有基础 CLI 全量纳入 agent contract。

### 影响范围

- 影响 agent-safe CLI discoverability；agent 可能退回读取文档、猜 hidden 命令或直接调用 Python 内部 API。
- 不影响 dashboard UI、store artifact 格式、job worker、rank board 或评估指标。
- 这不是模型能力问题，也不是 eval/codec/metric 误判。

### 修复方式

- 将 `init-run` 标记为稳定 mutating run 命令。
- 将 `validate-prediction` 标记为稳定只读 prediction 命令。
- 将 `process-next-job` 标记为稳定 mutating job 命令。
- CLI contract 测试检查三者出现在 `list-agent-commands`，并验证关键参数 schema、repeatable
  `--target-label`、positional prediction path 和 `--kind` 默认值。
- README、架构文档和脚本文档同步 agent 稳定命令面边界。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_cli.py -k 'agent_stable_commands or parser_commands'`

### 后续防线

- 新增 agent 可直接操作的 CLI 时，不能只加 parser/handler；必须同步 `AGENT_COMMAND_METADATA`，
  并让 `list-agent-commands` 输出足够的参数 schema，避免 agent 从自然语言 help 或源码猜调用形态。

## 2026-05-25: Eval Bench 首页再次退化成低价值状态墙

### 现象

总览页已经从旧 mini chart wall 收敛成 command deck，但第一屏仍把标题、统计块、下一步动作、管线、
活动矩阵和 readiness 分散展示。用户进入首页后仍需要自己判断“下一步该点哪里”，页面像状态字段展示，
而不是评测运营入口；同时整体交互反馈不够明显，hover 和状态流动感不足。

### 根因

上一版主要修复“信息太多”和“低价值面板回流”，但没有把首页的视觉优先级绑定到行动决策：下一步动作
只位于 focus panel 内部，顶部 hero 仍是静态摘要；通用 hover/transition 也只覆盖了部分按钮和总览卡片，
没有成为 dashboard 级交互层。

### 影响范围

- 影响 Eval Bench Dashboard 首页的第一屏判断效率、可点击性和实时感。
- 不影响 dashboard API、store schema、job/service 生命周期、rank board 排序或评估指标。
- 这不是模型能力问题，也不是 eval/codec/metric 误判。

### 修复方式

- 总览顶部改为优先动作 hero：标题直接表达当前状态，右侧大行动卡直接跳转到失败任务、待评估 run、
  队列、服务、排行榜或新建任务。
- 右侧 dock 只保留同步状态、四个紧凑运行态数值和三条高频入口，避免无意义面板占位。
- 主体 focus panel 改为“评测闭环”：四个可行动信号、四段管线和三泳道近期活动矩阵服务同一个判断。
- readiness 和最近产物保留在侧栏，作为可操作入口和最新产物索引，不承载二级诊断。
- 增加 dashboard 级 shared interaction layer，给 workspace card、表格行、chip、导航和标准命令补 hover、
  shimmer、lift 与 rail 动效；动效只表达可点击性和实时反馈，不改变数据语义。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 首页新增内容必须先回答“是否驱动下一步动作”；不能把 run manifest、artifact、parser、任务类型、
  模型分布或其它低频诊断字段搬回总览。
- 交互动效应落在可点击对象、状态胶囊、轨道和局部 hover 上；不能使用大面积装饰动画制造布局抖动。

## 2026-05-25: Eval Bench job 调度仍受固定首屏窗口影响

### 现象

前端目录页已经接入分页，但后端 job 生命周期仍存在固定窗口：dashboard 的 `/api/jobs/process-next`
只检查前 200 条 job，orchestrator 只扫描前 500 条 running job 和前 1000 条 queued job。当旧 running job
或 FIFO 中最早的 queued job 排在这个窗口之外时，调度器可能误判资源空闲或漏掉应优先启动的任务。

### 根因

`list_jobs(limit=...)` 同时承担 UI 列表分页和后端生命周期扫描两个职责。目录分页接入后，生命周期代码
仍复用面向页面的首屏窗口，没有单独的“完整匹配 job 集合”真源。

### 影响范围

- 影响 dashboard fallback worker 和 orchestrator 的 job claim / queue scan 正确性。
- 可能导致重复启动评测、跳过 FIFO 早期任务或错误释放 runtime/device/port 资源。
- 不影响评估指标、prediction parser、rank board 分数或模型能力；这是 dashboard/job lifecycle 误判。

### 修复方式

- `EvalBenchDatabase` 新增 `matching_jobs()`，作为不分页的完整匹配 job 真源。
- `job_page()` 继续负责 UI/API 分页，但复用 `matching_jobs()` 后再切片，避免过滤语义分叉。
- dashboard `/api/jobs/process-next` 和 orchestrator live-running / queued scan 改用 `matching_jobs()`。
- 增加回归：running job 超出前 200 条时 dashboard 不 claim 新任务；queued job 超出前 1000 条时
  orchestrator 仍按 FIFO 启动。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_dashboard.py -k 'process_next or live_job'`
- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_orchestrator.py -k 'queued_jobs_beyond_first_page or reserved or skips'`
- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_database.py`

### 后续防线

- UI/API 目录列表可以分页；job lifecycle、调度资源检查和 FIFO queue scan 不能使用固定 `limit` 首屏窗口。
- 新增 job lifecycle 入口时，先判断需要的是分页列表还是完整匹配集合；完整扫描必须走 `matching_jobs()`。

## 2026-05-25: Eval Bench 目录分页控件在六个页面重复实现

### 现象

Runs、Benchmarks、Jobs、Services、Compare 候选 run 和 Rank Board 都已经接入后端分页，但前端各页面仍各自
维护一份本地 pager 函数。每份函数都重复计算 start/end、previous/next offset 和按钮禁用状态。
后续如果其中一个页面改动分页边界，很容易造成禁用逻辑、offset clamp 或样式 className 分叉。

### 根因

分页能力是先按页面逐步补齐的：每补一个目录页面，就复制一份最小 pager。样本分页已经有
`samplePager.tsx`，但它只暴露 `SamplePager`，没有作为目录分页 primitive 使用。

### 影响范围

- 影响 Runs、Benchmarks、Jobs、Services、Compare 和 Rank Board 的前端分页一致性与维护成本。
- 不影响后端分页 API、查询语义、rank board 排序、comparison 计算或样本分页 URL 语义。

### 修复方式

- `samplePager.tsx` 新增 `PagerControl`，统一 range 展示、上一页/下一页按钮、禁用逻辑和可选 meta。
- `samplePager.tsx` 新增 `clampListPageOffset`，统一目录列表 offset 合法化。
- Runs、Benchmarks、Jobs、Services、Compare 和 Rank Board 删除本地 pager 函数，改为传入业务
  className、offset/limit/total 和必要 meta。
- UI contract 改为检查这些页面必须复用 `PagerControl` / `clampListPageOffset`，并禁止本地 pager
  函数回流。
- README 和架构文档同步目录分页真源。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增目录型页面时，分页只能复用 `PagerControl`；页面不应复制按钮、range 或 offset clamp。
- 如果未来需要页码跳转、每页大小切换或键盘分页，应先扩展 `PagerControl`，不要在单个页面局部实现。

## 2026-05-25: Eval Bench 窄屏 inspector split 把图片画布压缩到 2px

### 现象

Run/Benchmark inspector 在 desktop 下可用，但 compact / narrow / phone 宽度下，响应式规则把
`viewer-canvas-layout` 从横向 split 堆成单列；真实浏览器量测发现 `.image-stage` 高度只剩 2px，
对象检查器占满了可视化区域，用户看不到图片画布，表现为可视化面板“卡住”或无法完整显示。

### 根因

窄屏媒体查询统一把所有 `.resizable-split` 改成 `grid-template-columns: 1fr` 并隐藏 resizer，但
inspector 外层仍是固定一屏工作区和 `overflow: hidden`。`viewer-canvas-layout` 堆叠后没有显式纵向轨道，
CSS grid 在受限高度中把第一行画布压到最小，第二行对象检查器保留滚动高度。

### 影响范围

- 影响 Run Inspector 和 Benchmark Inspector 的 compact / narrow 视口可用性。
- 不影响后端 sample API、viewer 几何投影、pan/zoom 状态、label 过滤或评测指标语义。
- 这是前端 workspace layout 问题，不是模型能力或 eval/metric 误判。

### 修复方式

- 窄屏 `.visual-inspector-page` 改为局部滚动，避免外层 hidden 裁切堆叠后的检查器工作区。
- 窄屏 inspector split 改为自适应高度，侧栏限制在可滚动的紧凑高度内。
- 窄屏 viewer panel 和 `viewer-canvas-layout` 增加纵向轨道，保证 `.image-stage` 保留可操作高度，
  对象检查器继续在自己的面板内滚动。
- layout smoke 新增 inspector canvas pane 断言：有真实 inspector route 时，`.image-stage` 高度不能低于
  可操作阈值；如果 inspector 页面或对象检查器内容超出容器，必须由对应容器滚动。
- README 和架构文档同步窄屏 inspector 滚动边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 后续新增 resizable split 时，不能只改 columns；窄屏堆叠必须同时定义 rows、局部滚动容器和最低可操作高度。
- 视觉检查器的主画布必须在所有 layout smoke viewport 中保持可见，不能被对象检查器、样本列表或配置面板压缩为 0 高度。

## 2026-05-25: Eval Bench label 子任务缺少入口级非法 keypoint 回归

### 现象

Detection label 子任务已经有统一 policy、前端 detection-only 面板、preflight、init-run、worker、import 和
evaluator 适配，但 keypoint 非 `arrow` label 子任务的拒绝主要依赖语义 helper 单测和 CLI/preflight 用例。
如果后续有人改动 evaluator 或 prediction import 的调用方式，存在把非法 keypoint label 子任务绕过 UI 写入
评测链路的回归风险。

### 根因

`label_policy.py` 已经是真源，但入口级测试没有覆盖所有关键写入/评测入口。语义 helper 通过并不等于每个可被
agent 或脚本直接调用的入口都被锁住。

### 影响范围

- 影响评测标准一致性和 agent 直接操作 Eval Bench 的安全边界。
- 不代表模型能力问题；这是 eval label policy 防线覆盖不足，属于 eval/metric/data 语义边界风险。
- 不影响 detection label 子集计算、rank-board 排序、样本过滤或已有 report 读取。

### 修复方式

- 在 evaluator 回归中新增 `evaluate_run` 直接拒绝 keypoint 非 `arrow` `target_labels` 的用例。
- 在 prediction import 回归中新增导入入口拒绝 keypoint 非 `arrow` `target_labels` 的用例。
- README 明确 label 子任务的入口级回归覆盖 evaluator 与 prediction import，避免约束只停留在 UI 或单个
  helper。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_evaluator.py projects/eval_bench/tests/test_prediction_import.py -k 'target_labels or keypoint'`
- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_eval_semantics.py projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_prediction_import.py -k 'target_labels or keypoint'`
- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_cli.py -k 'target_labels or keypoint or resolve_target_labels or preflight'`
- `cd projects/eval_bench/frontend && npm run test:manifest-tools && npm run test:ui-contracts`

### 后续防线

- 新增或重命名 label 子任务入口时，必须同时覆盖 policy helper、CLI/preflight、worker/import/evaluator
  至少一个直接入口级测试。
- Keypoint 仍只允许 `arrow`；如果未来支持其它 keypoint label，必须先修改 `KEYPOINT_TARGET_LABELS`、文档、
  UI 暴露规则和所有入口级测试。

## 2026-05-25: Eval Bench Compare 候选 run 仍固定首屏 200 条

### 现象

Runs、Benchmarks、Jobs、Services 和 Rank Board 已经接入后端分页，但 Compare 页的候选 run 选择仍固定请求
`/api/runs?offset=0&limit=200`。当可对比 run 超过 200 时，对比工作区只能从首屏窗口里选择 baseline/candidate；
高级检索虽然复用 `/api/runs`，但筛选后不能继续完整翻页。

### 根因

Compare 页早期把 run 候选选择和对比报告加载绑定在同一个首屏列表里。后续 Runs 页已经补齐分页契约，但 Compare
没有同步收敛；同时 baseline/candidate 有 URL 状态，不能简单用“当前页是否包含已选 run”来判断选择是否有效。

### 影响范围

- 影响 Compare 页大量 run 下的完整候选浏览、筛选后翻页和复现实验对比选择。
- 不影响 comparison 计算、rank-board、run note、样本详情或 eval metric 语义。

### 修复方式

- Compare 页新增 `COMPARE_RUN_PAGE_SIZE=80`、`pageOffset` 和 `CompareRunPager`。
- 候选 run 请求 `/api/runs` 时传入 offset/limit；筛选条件变化回到第一页，数据减少时 clamp 到最后有效页。
- baseline/candidate 的有效值继续以已选 run id 为准；如果已选 run 不在当前页，下拉保留“已选择”占位项，
  并显示“当前页未加载该 run”，避免翻页误清空 URL 或上一页选择。
- layout smoke 要求 `/compare` 存在 `.compare-run-pager`。
- UI contract 锁住 Compare 页不能回退到固定 `limit: 200`。
- README 和架构文档同步 Compare 候选 run 必须后端分页，且翻页不能清空已选 run id。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 任何候选选择器如果背后是可增长目录，必须使用后端分页；不能把 select 的 options 规模当成固定小集合。
- 有 URL 或跨页选中状态的页面，分页只能改变候选窗口，不能把“当前页缺失”误判成“选择无效”。

## 2026-05-25: Eval Bench Jobs 高级检索仍固定首屏 200 条

### 现象

Runs、Benchmarks、Services 和 Rank Board 已经接入后端分页，但 Jobs 队列仍固定请求
`/api/jobs?limit=200`。当历史 job、失败重试和 preannotate/eval 任务记录超过 200 时，评测中心只能浏览首屏窗口；
状态、类型和全文高级检索虽然走后端过滤，但筛选后不能继续完整翻页。

### 根因

Job database、API 和 CLI 已经有 `job_page(offset, limit, kind, status, query)`，但前端队列页沿用了早期
“主页面先拉 200 条”的目录模式。结果列表分页契约已经在 Runs、Benchmarks、Services 和 Rank Board 中补齐，
Jobs 没有同步收敛。

### 影响范围

- 影响 Jobs 页大量任务记录下的完整浏览、筛选后翻页、删除后 offset 恢复和滚动稳定性。
- 不影响 job lifecycle、scheduler、runtime log、service registry 或 eval metric 语义。

### 修复方式

- Jobs 队列新增 `JOB_PAGE_SIZE=80`、`pageOffset` 和 `JobListPager`。
- 主队列请求 `/api/jobs` 时传入 offset/limit；筛选条件变化回到第一页，数据减少时 clamp 到最后有效页。
- compact 模式继续只取 12 条摘要，不改变总览或嵌入式轻量列表行为。
- layout smoke 要求 `/jobs` 存在 `.job-list-pager`。
- UI contract 锁住 Jobs 页不能回退到固定 `limit: 200`。
- README 和架构文档同步 Runs / Benchmarks / Jobs / Services / Rank Board 结果列表必须后端分页。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增目录型页面时，不要用固定首屏 slice 代替分页；高级检索、分页和 total 必须来自同一个后端查询。
- 页面筛选变化必须重置 offset，删除或数据减少后必须 clamp，避免“目录有数据但当前页空白”。

## 2026-05-25: Eval Bench Services 高级检索仍固定首屏 200 条

### 现象

Runs、Benchmarks 和 Rank Board 已经接入后端分页，但 Services 页仍固定请求
`/api/services?limit=200`。当服务登记、历史 stopped/failed 记录或外部 endpoint 变多时，模型服务目录只能看到首屏窗口；
高级检索虽然走后端过滤，但没有完整翻页能力。

### 根因

Service manager、API 和 CLI 已经有 `offset/limit`，但前端模型服务页沿用了早期“先拉 200 条”的目录模式。
分页契约先在结果库、基准集和排行榜补齐，Services 没有同步收敛到同一类后端分页结构。

### 影响范围

- 影响 Services 页在大量服务记录下的完整浏览、筛选后翻页和删除后 offset 恢复。
- 不影响 service registry、runtime 启停、health check、job 调度或 eval metric 语义。

### 修复方式

- Services 页新增 `SERVICE_PAGE_SIZE=80`、`pageOffset` 和 `ServiceListPager`。
- 请求 `/api/services` 时传入 offset/limit；筛选条件变化回到第一页，数据减少时 clamp 到最后有效页。
- layout smoke 要求 `/services` 存在 `.service-list-pager`。
- UI contract 锁住 Services 页不能回退到固定 `limit: 200`。
- README 和架构文档同步 Runs / Benchmarks / Services / Rank Board 结果列表必须后端分页。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增目录型页面时，不要用固定首屏 slice 代替分页；高级检索、分页和 total 必须来自同一个后端查询。
- 页面筛选变化必须重置 offset，删除或数据减少后必须 clamp，避免“目录有数据但当前页空白”。

## 2026-05-25: Eval Bench 总览 command deck 仍像低价值面板组

### 现象

总览页已经删除 mini chart wall，但左侧 focus panel 仍保留 Run/Ops/Volume 三组 track rail。用户进入首页后看到的
仍是状态面板集合，而不是可以立刻判断“当前该做什么”的控制台；页面虽然压缩了密度，但信息价值没有明显提升。

### 根因

上一轮只把低价值图表墙收敛成 command deck，没有继续筛掉对行动没有直接帮助的中间面板。Run/Ops/Volume 分组
仍在重复 Runs、Jobs、Services 页的目录信息，且布局上继续强化“面板墙”观感。

### 影响范围

- 影响 Dashboard 首页第一屏判断效率、视觉层级和交互质感。
- 不影响 eval metric、rank-board 计算、store 数据结构或 CLI 语义。

### 修复方式

- 总览左侧改为“当前决策”控制面：next action、报告覆盖、待评估、队列压力、在线服务四个可点击信号、
  四段评测管线和 Run/Job/Service 活动矩阵。
- 删除 `OverviewTrackGroup` 和 Run/Ops/Volume track 面板组，不再把总览当目录字段摘要。
- 总览布局改成两列 command surface，信号条最多四项，右侧只保留 readiness switchboard 和最近 run。
- 增强 overview 关键入口、按钮、状态 capsule 的 hover、active、focus、pulse 和 rail transition。
- README、架构文档、layout smoke 和 UI contract 同步新的总览准入边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 总览新增模块必须先回答是否能驱动下一步操作；不能把目录维度、低频排障字段或二级诊断面板搬回首页。
- 首页视觉最多保留四个一眼可读的关键信号；更多维度应进入对应工作区或高级筛选。
- 动效只用于强调可点击性、实时状态和状态转移，不得用装饰动画掩盖信息结构问题。

## 2026-05-25: Eval Bench Benchmarks 高级检索仍固定首屏 200 条

### 现象

Runs 和 Rank Board 已经接入后端分页，但 Benchmarks 页仍固定请求 `/api/benchmarks?offset=0&limit=200`。
当 benchmark 副本数量超过 200 时，基准集目录只能浏览首屏窗口；task/layer/split/query 高级检索虽然走后端，
但列表没有完整翻页能力。

### 根因

Benchmark store 和 CLI 已经有 `offset/limit`，但前端目录页沿用了早期“先拉 200 条”的简化模式。分页契约
先在 Rank Board 和 Runs 中补齐，Benchmarks 没有同步收敛。

### 影响范围

- 影响 Benchmark 目录在大量副本下的完整浏览、筛选后的可恢复翻页和滚动稳定性。
- 不影响 benchmark manifest、sample inspector、rank-board、run note 或 eval 语义。

### 修复方式

- Benchmarks 页新增 `BENCHMARK_PAGE_SIZE=80`、`pageOffset` 和 `BenchmarkListPager`。
- 请求 `/api/benchmarks` 时传入 offset/limit；筛选条件变化回到第一页，数据减少时 clamp 到最后有效页。
- layout smoke 要求 `/benchmarks` 存在 `.benchmark-list-pager`。
- UI contract 锁住 Benchmarks 页不能回退到固定 `limit: 200`。
- README 和架构文档同步 Runs / Benchmarks / Rank Board 结果列表必须后端分页。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增目录型页面时，不要用固定首屏 slice 代替分页；高级检索、分页和 total 必须来自同一个后端查询。
- 页面筛选变化必须重置 offset，删除或数据减少后必须 clamp，避免“目录有数据但当前页空白”。

## 2026-05-25: Eval Bench Runs 高级检索仍固定首屏 200 条

### 现象

Rank Board 已经使用 `/api/rank-board?offset&limit` 和独立 pager，但 Runs 页仍固定请求
`/api/runs?offset=0&limit=200`。当 run 数超过 200 时，结果库只能看到首屏窗口；高级检索虽然接入后端，
但没有完整分页浏览能力，筛选后的列表也不能在页面内继续翻页。

### 根因

早期 Runs 页把高级筛选和表格显示先接到同一份 `RunTable`，默认用固定 limit 简化列表；后续 Rank Board
已经补了分页契约，但 Runs 页没有同步收敛到同一类后端分页模式。

### 影响范围

- 影响 Runs 结果库在大量 run 下的完整浏览、筛选后的可恢复翻页和滚动稳定性。
- 不影响 run note 真源、rank-board 排序、evaluation 语义、label subtask 或后端 store schema。

### 修复方式

- Runs 页新增 `RUN_PAGE_SIZE=80`、`pageOffset` 和 `RunListPager`，请求 `/api/runs` 时传入 offset/limit。
- 筛选条件变化自动回到第一页；总数变化或删除 run 后，offset 会 clamp 到最后一个有效页。
- `RunTable` 增加 `footer` slot，让结果页分页控件留在统一表格栈内部，不复制表格实现。
- UI contract 锁住 Runs 页不能回退到固定 `limit: 200`；layout smoke 要求 `/runs` 存在 `.run-list-pager`。
- README 和架构文档同步 Runs / Rank Board 结果列表必须后端分页。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增结果列表页时，不要用固定首屏 slice 代替分页；高级检索、分页和 total 必须来自同一个后端查询。
- 任何筛选条件变化都要重置 offset；删除、归档或数据减少后要 clamp 到有效页，避免“有结果但当前页空白”。

## 2026-05-25: Eval Bench agent 命令发现缺少稳定调用形态

### 现象

`list-agent-commands` 已经能列出稳定命令、domain、是否修改状态、参数 schema 和互斥组，但没有输出每条命令的
稳定调用形态。agent 仍需要把 `scripts/eval_bench.py`、命令名和 usage 自己拼起来，容易重新退回从
自然语言 help 或当前进程名推断 argv 的模式。

### 根因

上一轮 agent CLI contract 重点锁住了 parser/handler/metadata 一致性和参数结构，但没有把“如何直接调用”
也纳入 machine-readable contract。`argparse.format_usage()` 本身还会携带当前进程名，在 pytest、stdin
或脚本运行时表现不同，不能直接作为稳定 agent 输出。

### 影响范围

- 影响 agent 发现命令后的可执行性和稳定性。
- 不影响 dashboard API、rank-board 排序、run note 写入、job lifecycle 或 store 文件格式。

### 修复方式

- `list-agent-commands` 顶层增加 `recommended_runner=[".venv/bin/python","scripts/eval_bench.py"]`。
- 每个稳定命令增加 `argv_prefix=["scripts/eval_bench.py", command]`。
- 每个稳定命令增加归一化单行 `usage`，固定以 `usage: eval_bench.py <command>` 开头，避免泄漏 pytest/stdin
  等当前进程名。
- `test_cli_lists_agent_stable_commands` 增加 recommended runner、argv prefix、stable usage 和 rank scheme
  usage 覆盖。
- README 和架构文档同步 agent CLI discovery contract。

### 回归测试

- `cd /home/tanjingyuan/code/arrow-vlm && .venv/bin/python -m pytest projects/eval_bench/tests/test_cli.py -q`
- `cd /home/tanjingyuan/code/arrow-vlm && .venv/bin/python scripts/eval_bench.py list-agent-commands`

### 后续防线

- 新增 agent 命令时，不能只暴露 parser/help；必须能从 `list-agent-commands` 得到可执行 argv 形态、参数结构和副作用标记。
- 不要直接把原始 argparse usage 暴露给 agent；先归一化，避免进程名随测试或调用方式漂移。

## 2026-05-25: Eval Bench Overview 右侧行动入口信息密度不足

### 现象

总览页已经从旧 mini chart wall 收敛为 command deck，但右侧“行动入口”仍只是 Rank Board / Create Job /
Services 三个普通链接。它能导航，但不能直接回答 service 是否在线、队列是否健康、是否还有待评估 run、
排行榜是否可用；最近 run 也直接取 `data.runs.slice(0, 6)`，隐含依赖 API 返回顺序。

### 根因

总览的信息架构只约束了“保留行动入口”，没有把行动入口定义成状态聚合组件。结果右侧空间承担了入口功能，
但没有承载“当前是否可用”和“下一步去哪”的组合判断。

### 影响范围

- 影响 Overview 首页的决策效率和视觉价值感。
- 不影响 rank board 计算、run note、job lifecycle、service health 或后端评估语义。

### 修复方式

- 将行动入口升级为 `OverviewReadinessPanel`：固定聚合 Services、Queue、Evaluation、Rank Board 四个状态入口。
- 每个入口展示状态 capsule、数量占比 rail、详情文案和目标路由，hover 动效沿用 overview command deck 体系。
- 顶部 console 增加动态 posture line，显示当前系统最重要的粗粒度状态。
- 最近 run 改为按 `created_at` 倒序截取，避免依赖 `/api/state` 返回顺序。
- README、架构文档和 layout/UI contract 同步记录 readiness switchboard 边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- Overview 新增模块必须继续回答“当前是否可用、卡在哪里、下一步去哪”，不要把低频诊断信息搬回首页。
- 行动入口不能回退成普通链接列表；必须保留状态、占比轨道和路由三类信息。
- 最近 run、最近 job 这类时间窗口必须显式排序，不依赖后端默认返回顺序。

## 2026-05-25: Eval Bench settings 快捷键捕获仍使用 raw button

### 现象

大部分页面标准动作已经收敛到 `ActionButton` / `CommandButton` / `IconActionButton`，但 settings 的快捷键捕获
控件仍在业务组件里直接写 `<button className="shortcut-capture">`。这不是阻塞式弹窗或表单提交问题，
但它绕过了统一按钮原语，后续 hover、disabled、键盘可访问性和视觉风格容易再次漂移。

### 根因

UI contract 只锁住了 shortcut reset/reset-all 使用 `ActionButton`，没有把 capture control 的按钮原语也纳入
契约。快捷键捕获虽然有专用样式和 `onKeyDown` 语义，但仍应该复用基础按钮组件。

### 影响范围

- 影响 Settings 页面快捷键映射控件的组件化一致性。
- 不影响 workspace settings schema、快捷键解析逻辑、viewer 行为或后端 Eval Bench 语义。

### 修复方式

- 将 shortcut capture 从 raw `<button>` 改为 `ActionButton variant="mini"`，保留 `shortcut-capture`
  class 和原有 `onKeyDown` 捕获逻辑。
- UI contract 增加静态防线：`settingsControls.tsx` 不能再出现 raw `<button>`，shortcut capture/reset/reset-all
  都必须通过 `ActionButton`。
- README 和架构文档补充快捷键捕获这类设置页交互也要复用按钮原语。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 设置页新增可点击控件时，除非是 `ui.tsx` 或 `controlPrimitives.tsx` 内部原语实现，否则不要直接写 raw button。
- 专用交互可以保留专用 class，但底层按钮语义仍优先复用 `ActionButton` / `IconActionButton`。

## 2026-05-25: Eval Bench agent CLI 发现面缺少参数 schema

### 现象

`list-agent-commands` 已能列出稳定命令、domain 和副作用标记，但 agent 仍只能从自然语言 help 推断参数。
例如 `preflight-job` 的 `--payload-json/--payload-file` 是必选互斥组，`rank-board` 的 weighted scheme
参数是可选互斥组，`import-predictions --target-label` 可重复；这些结构化参数形态没有机器可读出口。

### 根因

稳定命令 metadata 只覆盖命令级语义，没有复用 argparse parser 中已经存在的参数真源。agent 操作面仍要靠
help 文案猜 required、default、choices、repeatable 和 mutually exclusive group。这是 CLI agent contract
缺口，不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 agent 发现和调用 Eval Bench 稳定 CLI 的可靠性。
- 不改变任何命令参数、store 写入、dashboard API、job/service 生命周期或指标计算。

### 修复方式

- `list-agent-commands` 从当前 argparse parser 自动导出每个稳定命令的 `arguments` 和
  `mutually_exclusive_groups`。
- 每个 argument 暴露 `dest`、`flags`、`required`、`default`、`choices`、`type`、`action`、
  `repeatable`、`nargs`、`help` 和 `metavar`。
- CLI 合约测试锁住关键命令参数：`create-benchmark` 的 repeatable task、`preflight-job` 的必选互斥 payload、
  `rank-board` 的默认 F1 和 weighted scheme 互斥组、`import-predictions` 的 repeatable target label。
- README、架构文档和脚本文档同步 agent 参数 schema 约定。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py::test_cli_parser_commands_have_handlers_for_agent_contract projects/eval_bench/tests/test_cli.py::test_cli_lists_agent_stable_commands`
- `uv run python scripts/eval_bench.py list-agent-commands`

### 后续防线

- 新增稳定 agent 命令不允许只提供自然语言 help；参数必须来自 argparse 真源并由 `list-agent-commands` 暴露。
- 新增必选互斥组、choices 或 repeatable 参数时，应在 CLI 合约测试中补一个代表性断言。

## 2026-05-25: Eval Bench keypoint label 子任务可绕过 UI 创建

### 现象

Jobs 页已经只在 `task=detection` 时显示 label 子任务面板，agent 的 `resolve-target-labels` 也会对 keypoint 返回
`label_subtasks_supported=false`。但后端仍可能接受 keypoint manifest 或 CLI/import 输入中的
`target_labels=["icon"]` 这类非 arrow 标签，导致绕过 UI 后创建出不符合关键点评估边界的 run/job。

### 根因

此前 detection/keypoint 的“是否支持 label 子任务”主要停留在前端显示和 agent 查询 payload；真正的后端校验只检查
target label 是否存在于 benchmark label index，没有按 task 校验 keypoint 只能使用 `arrow`。这是 eval semantics /
label policy 边界缺口，不是模型能力问题。

### 影响范围

- 影响 keypoint job、init-run、prediction import、worker 执行和 evaluator 入口的 label scope 一致性。
- detection label 子任务能力不变，仍支持显式 label 子集。

### 修复方式

- `label_policy.py` 新增 keypoint task-level 校验真源：keypoint `target_labels` 只允许 `arrow`。
- `resolve_eval_semantics()`、`init-run`、prediction import 和 worker 都调用同一校验，避免不同入口绕过。
- job preflight 在 benchmark label index 校验前先做 task-level target label 校验；agent 查询也把该错误放入
  `resolve-target-labels` payload。
- CLI / eval semantics focused 测试覆盖 keypoint 非 arrow label 被拒绝。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_eval_semantics.py projects/eval_bench/tests/test_cli.py::test_cli_resolves_target_labels_for_agent_label_subtasks projects/eval_bench/tests/test_cli.py::test_cli_preflight_rejects_keypoint_label_subtasks projects/eval_bench/tests/test_cli.py::test_cli_preflight_rejects_unknown_target_label`
- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`

### 后续防线

- 任务级 label scope 规则只在 `label_policy.py` 增加，UI、worker、import 和 evaluator 不各自硬编码。
- 新增 task 子类型时必须同时定义是否支持 label 子任务，以及非法显式 label 的后端拒绝路径。

## 2026-05-25: Eval Bench 首页缺少可行动的下一步判断

### 现象

总览页已经删掉低价值 mini chart wall，但仍更像静态状态摘要。用户进入首页后不能立刻判断当前最该做什么：
是处理失败任务、评估已有预测、查看运行队列、启动服务，还是进入排行榜复盘。同时整体系统的 hover、
状态反馈和动效偏弱，显得像表格拼装而不是实时评测控制台。

### 根因

上一版总览只把面板数量压缩到 focus panel + recent run，没有建立“下一步动作”和“评测管线阶段”的信息结构。
布局测试也只检查 track rail 和活动矩阵是否存在，没有锁住可行动入口、管线阶段和交互动效。这是前端信息架构
和交互反馈问题，不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 Eval Bench Dashboard 首页的第一屏判断效率、可读性和系统交互质感。
- 不影响后端 store、rank board、evaluator、job/service 生命周期、comparison 或任何指标计算。

### 修复方式

- 首页改为控制台模型：顶部保留 Coverage、Pending、Queue、Services；主区域增加 `OverviewNextAction`，
  根据失败任务、待评估 run、活跃队列、服务状态和已评估 run 自动指向下一步页面。
- 增加四段 `OverviewPipeline`：Benchmarks、Predictions、Evaluated、Rank Ready，用 progress rail
  表示从数据到入榜的粗粒度进度。
- 保留 Run/Ops/Volume 三组运营压力 rail 和 Run/Job/Service 三泳道 12 日期桶活动矩阵；右侧改为行动入口
  加最近 run 紧凑摘要。
- 样式层补充首页入场、扫描光、卡片 hover、行动入口 hover、activity cell hover 等轻量动效，并把通用
  workspace/card/filter/service/settings 面板加入统一 hover 反馈。
- UI 合约和 layout smoke 改为检查 pipeline、next action、action panel、operational grid、3x12 活动矩阵、
  动效 transition，以及旧低价值面板不能回流。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- 首页新增模块必须能回答“当前是否可用、卡在哪里、下一步去哪”，否则放到 Runs、Inspector、Services、
  Rank Board 或 Compare。
- 总览测试继续禁止 precision / recall / IoU 细指标和低频排障面板回流；新增交互模块必须有 layout smoke
  或 UI contract 防线。
- 动效只服务状态反馈和可点击性，不用大面积装饰动画替代信息结构。

## 2026-05-25: Eval Bench agent CLI 缺少副作用元信息

### 现象

`list-agent-commands` 已经能列出稳定 agent CLI 命令，但输出只有 `name` 和 `help`。agent 可以发现命令，
却仍要从自然语言 help 猜测命令属于 benchmark、run、job、service 还是 comparison，也无法机器可读地区分
只读查询和会写 artifact、更新 SQLite 或启动/停止外部进程的操作。

### 根因

稳定命令集合只有 `AGENT_STABLE_COMMANDS` 一个平面 set，没有命令级 metadata 真源。副作用语义停留在文档和
命令帮助里，无法被测试锁住。这是 agent 操作合约问题，不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 agent 在自动编排 Eval Bench CLI 时的风险判断和命令选择。
- 不改变任何 CLI 参数、store 写入、dashboard API、rank board、job/service 生命周期或指标计算。

### 修复方式

- 新增 `AGENT_COMMAND_METADATA` 作为稳定 agent 命令真源，每条命令声明 `domain` 和 `mutates_state`。
- `AGENT_STABLE_COMMANDS` 改为由 metadata 派生，避免命令集合和 metadata 集合漂移。
- `list-agent-commands` 输出 `domain`、`mutates_state`、`mutating_count`、`read_only_count` 和 domain 列表。
- CLI 合约测试同时检查 metadata 覆盖、字段类型、读写计数和典型命令的副作用标记。
- README、架构文档和脚本文档同步说明：`mutates_state` 是副作用标记，不是权限控制。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run python scripts/eval_bench.py list-agent-commands`
- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests/test_cli.py`

### 后续防线

- 新增稳定 agent 命令必须先补 `AGENT_COMMAND_METADATA`，不能只补 `AGENT_STABLE_COMMANDS`。
- 会写 store、artifact、SQLite 或外部 service 进程状态的命令必须标记 `mutates_state=true`。
- agent-facing 文档不要把稳定命令描述成无副作用；稳定只表示接口可依赖。

## 2026-05-25: Eval Bench 总览仍在堆低价值状态面板

### 现象

总览页已经从 56+ mini chart 降到 5 个面板，但页面仍把 run 生命周期、数据规模、队列健康和服务运行态
拆成独立小面板。它们看起来更紧凑，却没有把用户真正需要的“评测推进、待评估 backlog、运行压力、最近 run”
组织成可直接判断下一步动作的控制台。

### 根因

上一轮约束仍以“保留 4-7 个高价值 mini chart”为目标，测试也在维护 chart matrix 和图表形态数量。
这会继续诱导实现为了满足面板数而把可由 track 或 rail 表达的状态拆成多个卡片。这是前端信息架构问题，
不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 Eval Bench Dashboard Overview 的第一屏判断效率和可读性。
- 不影响 store、rank board、evaluator、comparison、job/service 生命周期或指标计算。

### 修复方式

- 删除 Overview 的 `overviewCharts`、`OverviewMiniChartPanel` 和 `overview-chart-matrix` 渲染路径。
- 首页改为两列 command deck：左侧一个 focus panel 承载 coverage progress rail、Run/Ops/Volume track rail
  和 Run/Job/Service 三泳道活动矩阵；右侧只保留最近 run 紧凑摘要。
- 顶部只保留 Coverage、Pending、Queue、Services 四个可行动数字，避免 Bench/Runs/Done 这类重复汇总占位。
- UI 合约和 layout smoke 改为禁止旧 mini chart wall/chart matrix 回流，并检查 focus panel、recent panel、
  track rail 和 3x12 活动矩阵存在。
- README、架构文档和脚本文档同步更新为 command deck 约束。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 总览新增内容必须先归入评测推进、待评估 backlog、运行压力、数据/预测规模或最近 run 复盘之一。
- 不再以“面板数量”作为总览可视化目标；优先把同一判断链路合并到 track rail、progress rail 或活动矩阵。
- 排障型和维度分布型信息继续留在 Runs、Inspector、Rank Board、Services 或 Compare 页面。

## 2026-05-25: Eval Bench CLI 子命令分发缺少统一合约

### 现象

Eval Bench 已经补齐大量 agent-facing CLI，但 parser 定义和 `main()` 分发是两份手写列表。测试主要直接调用
`_cmd_*` 函数，无法证明 `scripts/eval_bench.py <command>` 真实入口一定能调到对应 handler。后续新增命令时，
很容易出现 help 中能看到命令，但入口分发遗漏的情况。

### 根因

CLI 子命令的真源不够集中：`_build_parser()` 维护命令名，`main()` 维护一条很长的 `if/elif` 分发链。
这不是模型能力问题，也不是 eval / codec / metric 误判，而是 agent 操作面的工程合约缺口。

### 影响范围

- 影响 Eval Bench agent CLI 的可维护性和新增命令可靠性。
- 不改变任何 job、service、run note、rank board、evaluator 或 dashboard API 语义。

### 修复方式

- 将 `main()` 分发收敛到 `_command_handlers()` 映射。
- 新增 `AGENT_STABLE_COMMANDS` 作为稳定 agent 命令集合真源，并通过 `list-agent-commands` 输出给 agent。
- 新增 CLI 合约测试：parser 暴露的命令集合必须等于 handler 映射集合。
- 同一测试锁住 `AGENT_STABLE_COMMANDS`，避免 dashboard-state、rank-board、run note、sample detail、comparison、
  service/job 生命周期等稳定入口被误删。
- 稳定命令面覆盖会改变 store 或外部进程状态的操作，例如 register/start/stop service、import/evaluate、
  compare 和 prompt template 写操作；名称不再使用 “safe”，避免误解为无副作用命令。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run python scripts/eval_bench.py list-agent-commands`
- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests/test_cli.py`

### 后续防线

- 新增 CLI 子命令时必须同时更新 parser 和 `_command_handlers()`；测试失败即表示真实入口不可执行。
- 新增稳定 agent 命令时必须同步更新 `AGENT_STABLE_COMMANDS`，使 `list-agent-commands` 成为 agent 可发现入口。
- agent-facing 命令不要只加内部 `_cmd_*` 单测，还要保证 parser/handler 合约覆盖到命令名。

## 2026-05-25: Eval Bench 总览页低价值面板回流

### 现象

总览页压缩后仍把 Notes、任务类型、模型分布、benchmark task、label footprint、样本/label 权重、
Job 日历和 scheduler 资源等二级诊断统计放进首页。用户在总控页看到很多面板，但这些面板不能直接支持
调度、数据规模判断、服务状态判断或最近 run 复盘，反而稀释了真正有价值的运行态信号。

### 根因

总览页的 smoke 只约束了“8-16 个图表”和图表形态，没有把首页信息价值作为边界约束。实现为了满足
密度和可视化数量，把可在 Runs、Inspector、Rank Board 或 Services 中查看的低频诊断维度也纳入总览。
这是前端信息架构问题，不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 Eval Bench dashboard 的 Overview 页面信息密度和可读性。
- 不影响后端 store、rank board、comparison、metric report 或 job/service 生命周期。

### 修复方式

- Overview 主图表收敛为 run 生命周期、评测覆盖、数据规模、队列健康和服务运行态。
- Signal deck 只保留 Benchmarks、Runs、GT samples、Predictions、Run/Job/Service 活动矩阵和实时遥测。
- 删除 Notes、任务类型、模型分布、benchmark task、label footprint、样本/label 权重、Job 日历和 scheduler
  资源在总览里的常显入口。
- Layout smoke 改为要求 4-7 个高价值 mini chart，并显式禁止这些低价值总览标题回流。
- 删除总览页已不再使用的 spark/mosaic 图表分支和 CSS，UI 合约改为检查当前真实保留的 ring、meter、
  cells、rails 四类图表，而不是检查陈旧函数是否存在。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增 Overview 面板前必须说明它能支持调度、数据规模、覆盖进度、队列/服务状态或最近 run 复盘中的哪一类操作。
- 排障型、归档型和模型/label 维度统计优先放在 Runs、Inspector、Rank Board 或 Services，不进入首页。
- Layout smoke 继续以禁止标题回流的方式防止低价值面板再次堆到总览。

## 2026-05-25: Eval Bench Rank Board 加权方案只在 CLI/API 可用

### 现象

Rank Board 后端 `/api/rank-board` 和 CLI `rank-board` 已经支持显式 `rank_scheme`，但前端独立排行榜页面
只能切换单一主指标，不能在页面内输入 weighted scheme，也不能查看每条 entry 的 score components。
这会让人类在核心排行榜工作台里仍要退回 CLI/API 才能复查显式加权排行。

### 根因

加权排行的真源已经放在 store/API/CLI，但前端只接了默认排序参数，没有把 `rank_scheme` 作为显式高级能力接入。
如果直接在前端计算加权分，又会形成第二套排行语义。这是展示/入口缺口，不是模型能力问题，也不是 eval /
codec / metric 误判。

### 影响范围

- 影响 Eval Bench dashboard 的 Rank Board 页面。
- 不影响后端 `rank_board` 计算、CLI `rank-board`、metric report 或 comparison report。

### 修复方式

- Rank Board 增加折叠式 `Weighted rank scheme` 面板，接受与 CLI/API 相同的 JSON。
- 前端只做基础 JSON 和字段校验，实际 weighted score、score formula、rank scheme 和 score components
  仍由 `/api/rank-board` 返回。
- 后端拒绝 scheme 时，错误显示在 weighted 面板内，表格继续保留上一份可用排行，避免核心页面整体失败。
- 加权模式下表格额外展示 `Weighted` 和 `Components` 列；默认模式仍保持 `f1_iou50` 主指标。
- Layout smoke 在真实 dashboard 上展开该面板、填入有效 scheme、启用加权排行，并断言 weighted chip
  和 weighted/components 表头出现；随后填入不支持的 metric，确认错误留在面板内且表格不消失。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- Rank Board 的默认主指标仍是 F1；任何综合分都必须由显式 `rank_scheme` 触发。
- 前端不能复制 weighted score 计算，只能把方案传给 API 并展示 API 返回的 `score_components`。
- Rank scheme API 错误必须是局部错误态，不能让独立排行榜主工作区整页失败。

## 2026-05-25: Eval Bench 标准动作按钮在业务页回流为原生 button

### 现象

Detection label 子任务面板和快捷键设置面板中，部分标准动作仍直接使用业务页原生 `button`：
`全部候选`、`默认策略`、自定义 label `添加`、单个快捷键 `重置` 和 `重置全部快捷键`。这会让按钮层级、
hover/disabled 状态和后续 dialog/action 规范继续出现双轨。

### 根因

早期页面在交互收口前先按局部样式补了按钮，后续虽然已经引入 `ActionButton`、`CommandButton`、
`IconActionButton` 和 `WorkspaceDialog`，但没有静态防线防止业务页继续保留标准动作按钮样式。这是前端
展示层组件边界问题，不涉及模型能力，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 Eval Bench dashboard 的 Jobs 新建评测弹窗和 Settings 快捷键设置面板。
- 不影响 job manifest、label policy、run note、Rank Board 或后端 CLI/API 行为。

### 修复方式

- Detection label 子任务的批量选择、默认策略和自定义 label 提交统一改用 `ActionButton variant="mini"`。
- 快捷键单项重置和重置全部统一改用 `ActionButton`，保留快捷键捕获按钮作为专用输入控件。
- 新增 `npm run test:ui-contracts`，静态阻止阻塞式浏览器弹窗、业务页自建 dialog shell、旧
  `sample-filters` 和这次已收敛标准动作回流。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增页面动作先使用 `ActionButton`、`CommandButton` 或 `IconActionButton`；只有样本行、画布 HUD、
  label chip、快捷键捕获这类具有独立输入语义的控件保留专用 button。
- 对 UI 边界的规则要进入可运行脚本，不只停留在 README 或架构说明里。

## 2026-05-25: Eval Bench 可视化检查器主统计条精细指标外露

### 现象

Run inspector 的样本卡片已经只显示 `真实 N / 预测 N`，但右侧 viewer 顶部的可见统计条仍直接展示
TP、FP、FN 和平均 IoU。用户在快速翻样本时会被精细评测指标干扰，主视图信息密度和“标注工具式检查”
定位不一致。

### 根因

`VisibleMetricStrip` 复用了 `viewerMetrics.ts` 的完整可见指标结果，并把所有字段都放在常显区域；
分 label 明细已经有折叠区承载 TP/FP/FN/P/R/IoU，但外层统计条没有按展示层级收敛。这是前端展示层级问题，
不是模型能力问题，也不是 eval / codec / metric 误判。

### 影响范围

- 影响 Eval Bench dashboard 的 Run inspector、工作台设置预览和成对样本 viewer 中复用的外层统计条。
- 不影响 `metrics.json`、comparison report、Rank Board 排序或对象级诊断数据。

### 修复方式

- `VisibleMetricStrip` 改为只渲染 `真实` 与 `预测` 两个紧凑计数块。
- TP、FP、FN、P/R、IoU 继续保留在折叠的分 label 明细和对象诊断里，避免丢失排障能力。
- layout smoke 增加 run inspector 断言：外层统计条必须只有两个 compact chip，不能出现 TP/FP/FN/IoU。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 可视化检查器主视图只放粗粒度状态；精细 metric 进入可折叠明细、排行榜或对比页。
- 新增 viewer 常显区域时必须先判断信息是否服务快速检查，而不是把 report 字段直接平铺出来。

## 2026-05-21: Eval Bench ephemeral vLLM TP 启动端口冲突

### 现象

Eval Bench ephemeral vLLM job 在服务健康检查前退出，worker 报：
`runtime process exited before ready: exitcode=1`。runtime 日志中 vLLM EngineCore 初始化失败，
TP worker 在 `tcp://127.0.0.1:10013` 建立 `TCPStore` 时报 `EADDRINUSE`。

### 根因

vLLM TP worker 会为内部分布式初始化随机选择本地通信端口。当前机器上随机选到的 `10013` 与已有端口
占用发生冲突，导致 rank0 的 TCPStore 无法监听，其他 rank 随后连接失败。这是 eval runtime 启动环境
问题，不是模型能力问题，也不是 codec / metric 误判。

### 影响范围

- 影响 Eval Bench 使用 `runtime.mode=ephemeral` 且 `tensor-parallel-size > 1` 启动本地 vLLM 的 job。
- 不影响已有 external/existing service 模式，也不影响已经生成的 prediction snapshot 和 metric 计算。
- 不影响 SFT 训练数据或训练主链。

### 修复方式

- `EvalBenchWorker.start_ephemeral_runtime` 在启动 vLLM 前自动设置 `VLLM_PORT`，由 API 端口派生独立内部
  端口段，例如 API `8000` 对应从 `28000` 开始查找可用端口。
- 保留用户显式设置的 `runtime.env.VLLM_PORT`，只在未设置时自动填充。

### 回归测试

- `.venv/bin/python -m pytest -q projects/eval_bench/tests/test_worker.py`
- `.venv/bin/python -m compileall -q projects/eval_bench/eval_bench scripts/eval_bench.py`

### 后续防线

- 新增本地多进程/多卡 runtime 时，不只检查 OpenAI API 端口，也要为后端内部通信端口预留稳定范围。
- Eval Bench 的 runtime 日志排障优先看第一个 `WorkerProc failed to start` 或 `EngineCore failed to start`
  的 root cause，避免只停留在 dashboard 外层的 `exitcode=1`。

## 2026-05-21: Banana Bench 隐式 PNG 图像路径导致 detection 评测失败

### 现象

`banana_bench` 上两个 detection job 在 inference 中途失败，报：
`No such file or directory: eval_bench_store/benchmarks/banana_bench/data/part2/images/prod_000876.png`。
实际 benchmark store 中存在的是 `prod_000876.jpg`。

### 根因

部分 benchmark JSON 没有显式 `image_path`。Eval Bench worker 对 `part*/json/*.json` 使用历史默认规则推断
同名 `.png` 图像，但 banana 原始 part2 图像包含 `.jpg`。这是 eval benchmark 数据路径标准不一致导致的
运行失败，不是模型能力问题，也不是 metric / codec 误判。

### 影响范围

- 影响共用 `banana_bench` 的 `grounding_arrow` 和 `grounding_layout` detection 评测。
- 不影响 `banana_point_arrow_bench`，该任务已经成功完成。
- 不影响训练数据和已完成的 point_arrow 评测结果。

### 修复方式

- 为 `eval_bench_store/benchmarks/banana_bench/data` 下 300 个 JSON 补充正确的 `image_path`，指向实际存在的
  `.jpg/.png` 图像。
- 新增 `projects/eval_bench/eval_bench/sample_paths.py` 作为 sample image / prediction JSON 路径映射的
  单一真源；worker、evaluator、prediction import、store 均复用这份 root-aware fallback。
- 新增 `projects/eval_bench/eval_bench/sample_scope.py` 作为 run sample target-label scope 的单一真源；
  run sample 的 GT、prediction、raw payload、prediction payload 和 diagnostics 都按 `target_labels` 展示。
- 当 JSON 未提供 `image_path` 时，按 `.png/.jpg/.jpeg/.webp` 顺序查找实际存在的同名图像，再回退到旧
  `.png` 规则。

### 回归测试

- `.venv/bin/python -m pytest -q projects/eval_bench/tests/test_evaluator.py projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py`
- `.venv/bin/python -m compileall -q projects/eval_bench/eval_bench scripts/eval_bench.py`
- 重新提交两个 detection job 后，二者均越过原失败样本位置继续 inference。

### 后续防线

- 创建 benchmark 时应尽量写入显式 `image_path`，不要依赖固定 `.png` 推断。
- 评测前的数据校验要覆盖“JSON image path 与真实图像后缀一致性”，尤其是混合 `.jpg/.png` 的 raw source。
- 后续新增路径兼容规则只能改 `sample_paths.py`，不能在 worker/evaluator/import/store 中继续复制私有 helper。
- 后续新增 run sample 展示范围或 diagnostics 兼容规则只能改 `sample_scope.py`；前端和 dashboard route
  只消费已经 scoped 的 API payload。

## 2026-05-19: Eval 可视化测试依赖已删除临时脚本

### 现象

快速回归 `uv run pytest -q` 在 collection 阶段失败，`tests/test_eval_common.py` 导入
`eval_common` 报 `ModuleNotFoundError`。该测试仍手动把 `scripts/tmp` 加入 `sys.path`，但
`scripts/tmp/eval_common.py` 已经不在当前仓库。

### 根因

离线 eval 预测结果转可视化快照的逻辑曾停留在临时脚本中；共享标注渲染已经进入
`src/shaft/metrics/visualization.py`，但 codec 结果到 boxes / keypoints / footer 的快照转换没有
同步迁入正式模块，测试也仍指向临时入口。

### 影响范围

影响默认快速回归的 collection，不影响模型能力，也不是 metric 计算或 codec 解析误判。风险在于 eval
可视化快照语义继续依赖 `scripts/tmp` 这类非稳定接口，后续迁移容易再次断链。

### 修复方式

- 新增 `src/shaft/metrics/prediction_visualization.py`，作为预测快照可视化的正式真源。
- `shaft.metrics` 导出 `render_prediction_visualization()`。
- 将测试重命名为 `tests/test_prediction_visualization.py`，不再修改 `sys.path` 或导入临时脚本。
- 更新 `docs/module_reference.md` 与 Eval Bench adapter 注释，移除已删除 `eval_common` 入口引用。

### 回归测试

- `uv run pytest -q tests/test_prediction_visualization.py`
- `uv run pytest -q`
- `uv run ruff check src/shaft/metrics/prediction_visualization.py src/shaft/metrics/__init__.py tests/test_prediction_visualization.py projects/eval_bench/eval_bench/adapters/__init__.py`
- `.venv/bin/python -m compileall -q src/shaft tests`

### 后续防线

- 测试不应把 `scripts/tmp` 加入导入路径；需要复用的 eval 能力必须先进入 `src/shaft` 正式模块。
- 临时离线脚本只允许编排，不应成为 codec / metric / visualization 共享语义的真源。
- 删除或迁移临时脚本时必须同时搜索测试、adapter 注释和模块文档中的旧入口引用。

## 2026-05-13: Eval Bench 子系统中间层不足导致语义漂移

### 现象

对子系统 review 后发现，Eval Bench 虽然已有 Control / Execution / Artifact 三层描述，但层级太粗。
target label scope、metric profile、job cancellation/resource lease 和 viewer/action 语义仍容易被
UI、worker、evaluator、import 和 comparison 各自推断。

### 根因

评估语义没有独立中间层，`task=detection` 被误用为 layout/arrow 子任务真源；job 生命周期缺少统一
resource lease 判断；metric profile 只是字符串字段，没有 registry 边界；pytest 直接跑
`projects/eval_bench/tests` 时也需要人工补 `PYTHONPATH`。

### 影响范围

影响 Eval Bench 的 evaluator、prediction import、comparison、orchestrator、dashboard fallback
worker 调度，以及后续新增任务类型、指标 profile、label scope、快捷键 action 和 viewer capability 的
扩展方式。

### 修复方式

- 新增 `eval_semantics.py`，统一解析 `task`、`metric_profile`、`target_labels` 和
  `target_labels_source`。
- 新增 `metric_profiles.py`，建立 `detection_iou_v1` 和 `keypoint_endpoint_v1` 的 profile registry。
- 新增 `metrics/` 包，把 matcher、sample diagnostic、geometry primitive 和 label aggregation 从
  `evaluator.py` 中拆出；`keypoint_endpoint_v1` 改为有序 endpoint distance matcher，bbox IoU 只作为
  诊断字段保留，不再决定 keypoint TP/FP/FN。
- Comparison report 和 Dashboard Compare 页保留 endpoint distance / endpoint pair delta，并把
  endpoint distance 下降作为 `keypoint_endpoint_v1` 的改善信号。
- 扩展 `label_policy.py`，返回 label 集合及来源，兼容旧 prompt ID 推断但明确标记为
  `legacy_prompt_id`。
- 新增 `job_lifecycle.py`，集中维护 job terminal/active/cancelled-resource lease 语义；取消请求后的
  live job 仍会占用 scheduler 资源。
- 更新 `docs/eval_bench_architecture.md`、`docs/architecture.md` 和 `projects/eval_bench/README.md`，
  把 Eval Bench 正式拆成七层。
- 在根目录 pytest 配置中加入 `projects/eval_bench` pythonpath，降低 focused test 入口成本。

### 回归测试

- `uv run pytest -q projects/eval_bench/tests/test_eval_semantics.py projects/eval_bench/tests/test_evaluator.py projects/eval_bench/tests/test_prediction_import.py`
- `uv run pytest -q projects/eval_bench/tests/test_eval_semantics.py projects/eval_bench/tests/test_evaluator.py projects/eval_bench/tests/test_prediction_import.py projects/eval_bench/tests/test_orchestrator.py projects/eval_bench/tests/test_dashboard.py`

### 后续防线

- 新增任务类型必须先进入 Evaluation Semantics Layer，再接 prompt、parser、metric 和 viewer。
- 新增 job 状态必须先进入 `job_lifecycle.py`，再接 database、scheduler、dashboard 和 status model。
- 新增 metric profile 时必须同时补 matcher 行为测试，防止 profile 只停留在字符串字段。

## 2026-05-13: Eval Bench running job 终止和 layout 指标作用域缺口

### 现象

任务中心只能取消 queued job，无法终止正在运行的评测；ephemeral vLLM runtime 一旦启动，用户只能等任务结束或手动查
PID。另一方面，在多任务 benchmark 上做 layout 检测时，如果 run spec 没有显式写 `target_labels`，
evaluator 会把同一张图里的 arrow GT / prediction 一起计入 detection 指标，导致 layout 指标被无关 arrow
样本拉低。

### 根因

Job 生命周期只覆盖了 `queued -> running -> succeeded/failed/cancelled` 的排队取消，没有给 running
job 设计取消请求、runtime 进程组终止和 worker 取消检查。指标侧把 `target_labels` 当成“已经写入 run
manifest 的字段”，没有在 evaluator 端兜底解析 prompt metadata 或内置 prompt ID，因此外部导入或旧 run
manifest 一旦缺失该字段，就回退成“评估所有 label”。

### 影响范围

- 影响 Dashboard Jobs 页、worker 执行中的 ephemeral runtime、外部 prediction snapshot 导入和
  `evaluate-run` 指标重算。
- 不影响训练主链，也不表示模型 layout 能力下降；这是 eval lifecycle 和 eval scope 的实现缺口。

### 修复方式

- `cancel_job` 支持 running job，写入 `cancel_requested`、取消时间和 cancelled progress metadata。
- Dashboard cancel endpoint 对带 `runtime_pid` 的 running job 尝试终止 ephemeral runtime 进程组。
- Worker 在解析、runtime ready 等待、逐样本推理、评估和最终落状态前检查取消请求；取消后的 job/run
  保持 `cancelled`，不会被后续异常覆盖成 `failed`。
- 新增 `label_policy.py`，集中实现目标标签解析：显式 `target_labels` 优先，其次 prompt metadata，
  最后按内置 prompt ID 推断 `grounding_layout.latest -> icon/image/shape`、arrow/keypoint prompt
  `-> arrow`。
- `import-predictions` 支持 `--target-label`，Dashboard 导入 prediction snapshot 也会传目标标签；
  evaluator 在旧 run 缺失 `target_labels` 时会按 prompt 补齐。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_evaluator.py projects/eval_bench/tests/test_prediction_import.py projects/eval_bench/tests/test_database.py projects/eval_bench/tests/test_worker.py projects/eval_bench/tests/test_dashboard.py`
- `PYTHONPATH=projects/eval_bench uv run python -m compileall projects/eval_bench/eval_bench`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:status-model`

### 后续防线

- running job 的终止必须同时覆盖状态、进程组、worker 检查点和 run manifest 状态，不能只改 UI 按钮。
- evaluator 不允许在多任务 benchmark 上静默“评估所有 label”；如果是 layout / arrow 子任务，必须有可追溯的
  target label 来源。
- 导入外部预测时必须保留评测作用域；`task=detection` 只表示 metric 类型，不表示 label 集合。

## 2026-05-13: Eval Bench 默认 CUDA 模板不应假设多卡

### 现象

默认 eval job manifest 和服务登记示例仍把 CUDA 写成 `0,2`，并把 vLLM
`tensor-parallel-size` 写成 `2`。这会让用户在普通单卡本地环境里创建任务后，默认配置就带有多卡假设。

### 根因

上一次修复只把非法的 `TP=3` 改成了可被 Qwen3VL attention heads 整除的 `TP=2`，但没有继续追问
“默认模板是否应该假设两张卡”。默认模板属于保守入口，不能把特定机器的多卡布局写成通用默认。

### 影响范围

- 影响 Dashboard 新建评测模板、CLI 文档示例和服务登记弹层默认值。
- 不影响用户手动声明的多卡 eval job、service registry 或 orchestrator 的多卡资源冲突检查。
- 这是默认配置语义问题，不是 vLLM、模型能力或 metric 问题。

### 修复方式

- 内置 `eval_job / layout_eval_job / keypoint_eval_job` 共用的 manifest helper 改为
  `CUDA_VISIBLE_DEVICES=0`、`tensor-parallel-size=1`。
- Services 页登记本地 vLLM 的 CUDA 默认输入改为 `0`，TP 默认继续为 `1`。
- `projects/eval_bench/README.md` 和 `docs/scripts.md` 的 create-job / register-service 示例同步改为单卡默认。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_job_spec.py`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:dialogs`
- Playwright 检查 Services 登记弹层：CUDA 输入为 `0`，TP 输入为 `1`。

### 后续防线

- 默认模板只能使用单卡保守值；多卡、跨卡 ID 和 TP > 1 必须由用户显式修改。
- 文档示例不得把本机 GPU 拓扑写成默认推荐；可以在说明里另列多卡覆盖方式。
- 每次修改 job template 的 runtime 默认值，必须同步检查 Dashboard 表单默认值、CLI 示例和 job spec 测试。

## 2026-05-13: Eval Bench 按钮对比度和弹层表单宽度不稳定

### 现象

部分按钮在浅色/深色背景上的文字对比度不足；弹层里的表单仍沿用固定列宽，长路径、URL、manifest
相关字段容易被压窄，错误/结果提示也会挤在普通字段列里。

### 根因

按钮只继承了通用 `primary/secondary` 样式，没有明确文本颜色、图标承载方式和主次层级。
弹层表单复用了早期页面内表单的固定列布局，没有按字段类型规划宽度。

### 影响范围

- 影响 Eval Bench dashboard 的 Jobs、Benchmarks、Runs、Services 和 Settings 页面。
- 不影响后端评测语义、指标计算或训练主链。

### 修复方式

- 使用 `image_gen` 生成 2x4 表单动作图标母版，透明化后裁剪为 restore/apply/preflight/enqueue/create/save/reset/clear
  8 个 PNG，并接入 `APP_ICON_PATHS`。
- 重设 primary/secondary/mini/settings action button 颜色、hover、图标背景 tile 和 disabled 状态。
- 收窄 `.page-command-row` 的说明文字选择器，避免它覆盖 command button 内部文字颜色。
- 将普通弹层表单改为 12 列规划：普通字段 4 列、长字段 6 列、结果/错误全宽、窄屏全宽。
- 给创建、导入、保存、预检查、入队和设置重置按钮补齐动作图标。
- 在 `docs/eval_bench_ui_icon_design.md` 中记录按钮层级、图标使用边界和弹层表单宽度规则。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:dialogs`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/jobs npm run render-check`

### 后续防线

- 新增表单字段必须判断字段长度；路径、URL、长 ID、结果和错误提示不能使用普通短字段宽度。
- 新增主要动作按钮优先使用 `APP_ICON_PATHS` 中的动作图标；没有合适图标时先扩展图标库再接入。
- 任何深色按钮必须显式声明高对比文本色，不能依赖继承。

## 2026-05-13: Eval Bench 图标体系缺少业务语义边界

### 现象

Dashboard 导航、总览指标和主操作按钮都复用通用矢量图标；logo 也是单一图片资源，没有形成可维护的
业务图标库。后续若继续给按钮临时堆图标，页面会重新回到“组件堆砌但缺少层级”的状态。

### 根因

前端此前只区分“有无图标”，没有区分业务语义图标和基础工具图标，也没有图标路径映射真源。
图标资产、UI 层级和使用边界没有进入开发文档。

### 影响范围

- 影响 Eval Bench dashboard 的品牌识别、导航扫描和主操作识别。
- 不影响 Eval Bench 后端、评测指标、worker 或训练主链。

### 修复方式

- 使用 `image_gen` 生成 4x4 业务图标母版，做 chroma-key 透明化后裁剪成 16 个 256x256 PNG。
- 新增 `frontend/src/iconLibrary.tsx` 作为图标路径唯一映射。
- 将 sidebar logo、导航、总览指标和主操作按钮接入 PNG 图标库。
- 保留关闭、删除、搜索、归档等基础工具动作为矢量图标，避免小尺寸 PNG 误用。
- Dashboard 后端显式挂载构建产物中的 `/icons`，并为 `/logo.png` 提供静态文件响应，避免 SPA
  fallback 把 PNG 请求返回成 `index.html`。
- 新增 `docs/eval_bench_ui_icon_design.md` 记录 UI 压缩原则、图标库存放路径、使用边界和生成 prompt。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/jobs npm run render-check`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:dialogs`
- `cd projects/eval_bench/frontend && node --input-type=module <broken-image-check>`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_serves_spa_fallback_when_frontend_is_built`

### 后续防线

- 新业务图标必须先进入 `APP_ICON_PATHS`，页面组件不得硬编码 `/icons/...`。
- 只把业务语义图标做成 PNG；通用工具动作继续用矢量图标。
- 生成图标不得包含文字，所有可读文案都由前端 HTML 渲染。

## 2026-05-13: Eval Bench 主页面表单堆叠导致工作台密度下降

### 现象

Jobs、Benchmarks、Runs 和 Services 页面把低频操作放在嵌套 tab 或折叠面板里，用户需要在同一主页面里
面对表格、队列、表单、按钮和预检查结果。设置页仍有横向 slider，占用宽度但不适合精确配置。

### 根因

早期 dashboard 把“目录页”和“创建/导入/登记页”合并成一个 WorkspaceTabs 结构，认为折叠即可减少干扰。
实际使用中这仍然让低频配置进入主信息架构，形成页面嵌套和按钮堆叠。设置页沿用了演示式 slider，而不是
工程设置更需要的紧凑数值输入。

### 影响范围

- 影响 Eval Bench dashboard 的 Jobs、Benchmarks、Runs、Services 和 Settings 页面体验。
- 不影响 Eval Bench 后端 artifact、worker、metric 或 Shaft 训练主链。
- 属于 dashboard 信息架构和交互密度问题，不是 eval 语义或模型能力问题。

### 修复方式

- 移除主页面嵌套 tabs，删除 `@radix-ui/react-tabs` 前端依赖。
- 新增统一 `WorkspaceDialog`，将新建评测、创建 benchmark、导入 prediction snapshot 和登记 service
  收进临时弹层；主页面只保留队列、目录、最近结果和服务状态。
- Jobs 页不再内嵌完整结果库表格，只保留最近结果入口；完整 run 管理仍在 Runs 页面。
- `StyleSlider` 改为紧凑 `number` input，设置页不再使用横向 slider。
- 新增 `npm run test:dialogs`，用真实浏览器覆盖四个临时弹层的打开、表单渲染和 Escape 关闭。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:manifest-tools`
- `cd projects/eval_bench/frontend && npm run test:workspace-settings`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:dialogs`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/settings npm run test:settings-preview`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766 npm run test:shortcuts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/benchmarks npm run render-check`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/runs npm run render-check`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/services npm run render-check`

### 后续防线

- 主页面只承载高频浏览、检视和排障，不承载常驻创建/导入表单。
- 新增低频操作优先使用临时弹层；如果操作复杂到需要长期上下文，再规划独立 route，不放入嵌套 tab。
- 工程设置优先使用可精确输入、可压缩排列的控件；避免横向 slider 和大块表单控件占据主画布空间。

## 2026-05-12: Eval Bench 默认 job 仍跑 layout prompt 导致 arrow 预测缺失

### 现象

在 `multitask_val_v1` 上新建 eval job 后，用户预期检查 arrow 检测结果，但落盘 run 的
prediction/report 仍主要是 `icon / image / shape`，几乎没有 `arrow`。当前本地 run
`eval_20260512_161233_83684bf1` 的 run manifest 记录的是 `prompt_id=grounding_layout.latest`，
`target_labels=["icon","image","shape"]`，并不是箭头检测任务。

### 根因

这不是模型在 arrow 任务上直接失败，而是 Eval Bench job 默认模板和 prompt 切换语义有偏差：

- 默认 `eval_job` 仍使用 layout prompt，虽然默认模型路径已经指向 arrow/layout/keypoint 混合模型。
- 旧 SQLite prompt template seed 使用 `INSERT OR IGNORE`，内置 prompt 后续新增的 `target_labels`
  不会刷新到既有 `eval_bench_store/db/eval_bench.sqlite`。
- 前端应用 prompt 时只在 prompt 声明 target label 时写入 `target_labels`，没有清理旧 manifest 残留。
- prediction parser 只接受 `arrow`，没有把模型可能输出的 `single_arrow / double_arrow` 归一为
  benchmark 使用的 `arrow` 标签。

### 影响范围

- 影响 Eval Bench 新建 job、preflight、worker 生成 run manifest、report label 过滤和 dashboard 检视。
- 不影响 benchmark GT copy，也不影响 Shaft 训练主链。
- 这是 eval job / parser / prompt registry 的语义问题，不应直接归因于模型能力。

### 修复方式

- 默认 `eval_job` 改为 `grounding_arrow.latest` + `target_labels=["arrow"]`。
- 保留单独的 `layout_eval_job` 和 `keypoint_eval_job`，避免把 layout、arrow detection、arrow keypoint
  混在一个模板里。
- repo 内置 prompt template seed 对 `metadata.source=repo_config` 的旧记录执行刷新，但不覆盖 dashboard
  保存过的自定义 prompt。
- 前端应用 prompt template 时同步写入 target labels；如果 prompt 没有声明 target labels，则清空旧
  manifest 上残留的 target labels。
- detection parser 将 `single_arrow / double_arrow / arrow_instance / arrows` 归一为 `arrow`，并在
  `extra.source_label_before_normalization` 中保留原始标签。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_prediction_parser.py projects/eval_bench/tests/test_database.py`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests`
- `cd projects/eval_bench/frontend && npm run test:manifest-tools`
- `cd projects/eval_bench/frontend && npm run build`
- `python -m compileall projects/eval_bench/eval_bench`

### 后续防线

- 多任务 benchmark 上一次 run 的评估标签必须由 prompt/template 明确声明，不能只看 `task=detection`。
- 新增或修改内置 prompt template 时必须验证旧 SQLite store 能刷新 repo 内置元数据。
- 前端切换 prompt 必须清理与旧 prompt 绑定的配置残留，尤其是 target labels、parser 和 metric profile。
- parser 要接受模型可能输出的任务内细分类，并映射到 benchmark 的 canonical label。

## 2026-05-12: Eval Bench 原图直出导致检视首屏资源过重

### 现象

即使前端限制预加载半径，viewer 首屏仍然直接使用 `/image` 原图。对于 4K、多 MB PNG，浏览器仍需下载和解码
完整图片后才能显示画布，影响样本检视首屏响应。

### 根因

Eval Bench 后端只有原图 FileResponse，没有面向 dashboard 检视的派生图层。前端无法区分“复盘证据原图”和
“交互检视底图”，只能把原图同时作为证据文件和 viewer display source。

### 影响范围

- 影响 benchmark/run/comparison/settings preview 的图片检视首屏体感。
- 不影响 benchmark copy、run prediction、metric 计算或原始证据文件。
- 不属于模型能力问题，是 dashboard 图像资源分层缺失。

### 修复方式

- sample payload 保留 `image_url` 原图，同时新增：
  - `image_preview_url`: `/image/preview?max_side=1800`，服务端生成并缓存 JPEG 缩略代理。
  - `image_tile_url_template`: `/image/tiles/{level}/{x}/{y}`，服务端按 level/x/y 生成并缓存 JPEG 金字塔瓦片。
  - `image_tile_size`: 当前瓦片边长，默认 512。
- benchmark、run、settings preview 和 comparison sample 都走同一套 image URL payload。
- viewer 和预加载默认使用 `image_preview_url`，原图仍保留在 API 中用于证据复核。
- viewer 高倍缩放超过阈值并停顿后，按 `image_tile_url_template` 延迟加载少量金字塔瓦片；瓦片数量受上限保护，
  不在首屏或连续滚轮过程中抢占资源。
- 派生图缓存写入 `eval_bench_store/cache/image_proxy/`，缓存 key 包含源图路径、mtime、size 和派生参数。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py`
- `python -m compileall projects/eval_bench/eval_bench`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- dashboard viewer 不应直接把多 MB 原图作为默认 display source；原图接口用于证据，检视默认走 preview proxy。
- 金字塔瓦片必须延迟到用户高倍缩放并停顿后加载，并限制可见瓦片数量，避免把原图解码压力从首屏转移到滚轮事件。

## 2026-05-12: Eval Bench 样本检视预加载过重和文本过度截断

### 现象

Eval Bench dashboard 中多处 run id、路径、样本文件名、表格单元格和设置项文本显示为省略号，影响排障时
直接读取完整信息。部分样本进入图片可视化检视时出现明显等待，用户体感可到几十秒。

### 根因

- 文本层面：前端为了保护表格、卡片和工具栏布局，多个公共选择器使用 `overflow: hidden`、
  `text-overflow: ellipsis` 和 `white-space: nowrap`，但没有区分“短标签控件”和“需要完整读取的工程信息”。
- 性能层面：后端 sample detail API 和图片 FileResponse 本身不是瓶颈。本地实测：
  - `/api/runs/config_smoke_prompt_params/samples/0` 约 3ms。
  - `/api/runs/config_smoke_prompt_params/samples/0/image` 约 30ms，图片大小约 7MB。
  - 当前 benchmark 中最大 PNG 约 11MB，分辨率可达 `4096x2234`。
- 真实瓶颈是前端 `preloadSampleImages` 每次选中样本时预加载前后 4 个样本并包含当前样本，最多会并发触发 9 张
  高分辨率 PNG 的本地 HTTP 传输和浏览器解码；这仍然会占用浏览器解码、内存和本机 IO，不是外网带宽问题。

### 影响范围

- 影响 Eval Bench dashboard 的可读性和样本检视交互延迟。
- 不影响模型评测结果、metric 计算或 prediction artifact。
- 不属于模型能力问题，是 dashboard 前端资源调度和信息密度设计问题。

### 修复方式

- 文本显示改为默认允许换行和 `overflow-wrap: anywhere`，对 run id、路径、样本名、表格单元格、设置项、
  service log path、shortcut action 等工程信息不再默认省略。
- 图片预加载从“当前样本前后 4 个”收敛为“空闲时只预加载相邻样本”，并排除当前样本，避免和主 `<img>` 请求重复。
- 为预加载增加简单 URL cache 和 effect cleanup，快速切换样本时取消尚未执行的 idle preload。
- sample detail prefetch 从前后 3 个收敛为前后 1 个，减少后台 JSON 请求。
- `test:viewer-performance` 增加初始检视图片请求数量断言，防止回退到一次打开多张大图。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:status-model`
- `cd projects/eval_bench/frontend && npm run test:workspace-settings`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/settings npm run test:settings-preview`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:shortcuts`
- `cd projects/eval_bench/frontend && npm run test:viewer-performance`
  - `canvas_renders_during_pan_zoom = 3`
  - `gt_layer_renders_during_pan_zoom = 0`
  - `pred_layer_renders_during_pan_zoom = 0`
  - `image_requests_during_initial_inspection = 2`
- `git diff --check`

### 后续防线

- 新增样本检视预加载时必须限制并发和半径，并优先使用 idle 调度。
- 对 4K / 多 MB 图片，不能把“本地浏览器计算”误认为没有成本；浏览器解码、内存和本地 HTTP 传输仍是交互路径的一部分。
- 工程信息字段默认应完整可读；只有固定宽度控件、图标按钮或明确需要单行的短标签可以使用省略号。

## 2026-05-12: Eval Bench 前端功能边界集中在 main.tsx

### 现象

Eval Bench dashboard 多轮迭代后，服务页、设置页控件、viewer pan/zoom、SVG 叠图、对象列表、快捷键、
manifest 转换、样本导航和状态规则都堆在 `frontend/src/main.tsx` 附近。虽然功能可运行，但继续新增
图层、对象类型、服务动作或快捷键时，会让路由文件承担过多职责，也容易留下重复状态源。

### 根因

前端早期以快速打通页面为目标，先在页面组件里就地实现业务规则和局部 UI。后续修复颜色、快捷键、
滚轮性能和服务状态时，如果只继续在原文件里追加 patch，就会形成“页面即模块”的结构，无法表达
viewer、settings、services、manifest、navigation、status 这些实际边界。

### 影响范围

- 影响 Eval Bench dashboard 的可维护性和后续扩展成本。
- 不影响后端 metric、prediction artifact、job 执行和模型评测结果。
- 不属于模型能力问题，是 dashboard 前端代码组织和状态真源边界问题。

### 修复方式

- 把 `main.tsx` 收敛为路由和页面装配层，禁止它继续承载新的跨页面业务规则。
- 新增/整理前端功能模块：`statusModel.ts`、`workspaceSettings.ts`、`viewerCanvas.tsx`、
  `viewerPanels.tsx`、`viewerGeometry.ts`、`viewerMetrics.ts`、`settingsControls.tsx`、
  `servicesPage.tsx`、`manifestTools.ts`、`sampleNavigation.ts`、`formatters.ts`、
  `controlPrimitives.tsx`、`workspaceLayout.tsx`、`dashboardState.ts`、`jobsPage.tsx`、
  `runTables.tsx`、`filterControls.tsx`。
- workspace split pane 的 resize、尺寸恢复和 localStorage 持久化落在 `workspaceLayout.tsx`，页面组件只消费布局能力。
- 评测中心、job queue、manifest 预检查、runtime log 和 run/benchmark 表格从路由文件拆到 `jobsPage.tsx`
  与 `runTables.tsx`；`main.tsx` 不再直接承载 job 操作 mutation 或 run 表格操作 mutation。
- viewer 高频渲染边界单独落在 `viewerCanvas.tsx`，`CanvasStage` 和 SVG instance layer 不再混在路由文件里。
- viewer 操作面板、对象列表、可见指标和实例统计落在 `viewerPanels.tsx`，设置页只复用基础控件，不反向依赖 viewer。
- 设置页 section、preference row、label 颜色添加和快捷键编辑器落在 `settingsControls.tsx`。
- 服务页的查询、mutation、日志和表单落在 `servicesPage.tsx`。
- 快捷键覆盖脚本改为扫描 `main.tsx`、`viewerCanvas.tsx`、`viewerPanels.tsx` 和 `settingsControls.tsx`，
  避免模块拆分后漏检新的全局 keyboard entry。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:status-model`
- `cd projects/eval_bench/frontend && npm run test:workspace-settings`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/settings npm run test:settings-preview`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:shortcuts`
  - `static_actions_checked = 9`
  - `benchmark_sample_navigation = true`
  - `run_viewer_actions = true`
  - `comparison_viewer_actions = true`
  - `settings_keymap_editor = true`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/runs/config_smoke_prompt_params npm run test:viewer-performance`
  - `canvas_renders_during_pan_zoom = 3`
  - `gt_layer_renders_during_pan_zoom = 0`
  - `pred_layer_renders_during_pan_zoom = 0`
- `git diff --check`

### 后续防线

- 新增 Eval Bench 前端能力时先确认真源模块；跨页面规则不得直接写进 `main.tsx`。
- 新增 viewer 渲染能力优先放到 `viewerCanvas.tsx` 或 `viewerGeometry.ts`；新增 viewer 操作 UI 放到
  `viewerPanels.tsx`；新增设置 schema 放到 `workspaceSettings.ts` 并同步 settings UI。
- 新增全局快捷键时必须扩展 `SHORTCUT_ACTIONS` 和 `test:shortcuts` 的扫描/浏览器行为覆盖。
- 允许继续拆分 `main.tsx` 中的页面级 orchestration，但必须保持 API、状态、viewer、settings 和
  manifest 这些模块边界不漂移。

## 2026-05-12: Eval Bench 业务状态和动作权限散落在组件里

### 现象

Job、run、service 的状态 badge 文案和按钮启用条件分别散落在 `ui.tsx`、`main.tsx` 的多个页面组件中。
同一个状态在不同页面容易显示成不同语义，running job / service 这类活跃对象也缺少统一 live 反馈。

### 根因

前端只有 API record 的 `status` 字符串，没有独立的 dashboard 状态模型。组件直接写
`job.status === "queued"`、`service.status === "running"` 这类判断，导致状态文案、视觉 tone、
可执行动作和后端状态机没有稳定的前端真源。

### 影响范围

- 影响 Eval Bench Jobs、Runs、Services 和 Overview 中的状态展示与操作按钮。
- 不影响后端 job/service 状态机，也不影响 metric 计算。
- 不属于模型能力问题，是 dashboard 前端状态表达层级错误。

### 修复方式

- 新增 `projects/eval_bench/frontend/src/statusModel.ts`，集中维护 `job / run / service` 的状态文案、
  tone、phase、live 标记和动作权限。
- `Badge` 改为读取 `statusModel.ts`，支持 domain override；service running 显示为“服务就绪”，run imported
  显示为“待评估”。
- Jobs、Runs、Services 页按钮启用条件改用 `canCancelJob()`、`canDeleteJob()`、`canEvaluateRun()`、
  `canArchiveRun()`、`canStartService()`、`canStopService()` 和 `canDeleteService()`。
- badge 增加 `warning / info / muted` tone 和 live pulse 动画；按钮禁用态统一收敛到视觉设计层。

### 回归测试

- `cd projects/eval_bench/frontend && npm run test:status-model`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 新增业务状态或按钮权限时先扩展 `statusModel.ts`，不要在页面组件里新增平行判断。
- `Badge` 只能消费状态模型；状态文案不再在业务组件里散落维护。
- 对会启动、停止、删除或归档资源的按钮，必须在 `test:status-model` 中覆盖启用条件。

## 2026-05-12: Eval Bench 设置页叠图预览颜色不跟随和 label 颜色大小写问题

### 现象

工作台设置中的叠图预览修改颜色后没有直观实时变化；同时 label 颜色匹配和大小写展示语义不清晰，
容易让用户误以为 label 颜色规则是大小写敏感的，或者 UI 会把 label 强制显示成大写。

### 根因

`useWorkspaceSettings()` 之前会为每个运行时 label 自动生成 fallback 颜色，并把这些 fallback 当作
`labelColors` 传给 viewer。`resolveInstanceColor()` 又优先使用 `labelColors[label]`，导致全局 GT / Pred
颜色被自动 label 色覆盖。设置页虽然 state 已更新，但图上的实例仍使用 fallback label 色，看起来像预览没更新。

此外，label 颜色规则直接用原始字符串作为 key；`Arrow`、`arrow`、`ARROW` 会被当作不同规则。部分控制项
CSS 也使用 `text-transform: uppercase`，不适合展示真实任务 label。

后续排查还发现一层 CSS 优先级问题：实例 `<g>` 虽然已经写入 `--instance-color`，但更高优先级的
`.overlay-instance.gt.*` / `.overlay-instance.pred.*` 规则仍直接读取 `--overlay-gt`、`--overlay-pred`
等全局变量，低优先级的通用 `--instance-color` 规则无法覆盖它们。因此 React state 和 inline CSS 变量
已经更新时，SVG 里的 rect/polyline/text/circle 仍可能沿用旧色或默认色。

### 影响范围

- 影响 Eval Bench 工作台设置页叠图预览。
- 影响 run inspector、benchmark viewer 和 compare 中 label 自定义颜色的一致性。
- 不影响后端 metric、prediction snapshot、job 执行和评测结果。

### 修复方式

- label 颜色只保存用户显式设置，不再把自动 fallback 色作为真正的 label override 传给 viewer。
- label 颜色配置改成 `label × role` 矩阵，role 包括 `GT`、`Pred`、`FN`、`FP`；例如
  `arrow × GT` 与 `arrow × Pred` 是两个独立颜色单元格，不再用单一 label 色覆盖所有 role。
- label 匹配使用 `label.trim().toLowerCase()` 作为内部 key，因此大小写不敏感。
- UI 展示继续保留数据里的原始 label 文案，不为了匹配逻辑强制改成大写或小写。
- 没有显式 `label × role` 颜色时，viewer 回退到 GT / Pred / FP / FN 固定 role 默认颜色；外观页不再提供全局图层颜色入口，避免重新形成“role 色”和“label 色”两套用户配置层级。
- SVG 状态规则统一改为读取 `--instance-color`，并补齐 `--overlay-gt`、`--overlay-pred`、`--overlay-fn`
  和 `--overlay-fp` CSS 变量，避免状态选择器绕过 viewer 的颜色真源。
- overlay style、interaction slider 和预测线型选项收敛到 `workspaceSettings.ts` 的配置 schema；
  Run inspector 控件、Settings 页面控件和 normalizer 都读取同一份范围/step/scale 定义，避免三处各自维护。
- 快捷键从页面硬编码文案改为 `SHORTCUT_ACTIONS` action registry；Settings 页展示 action、键位和冲突状态，运行时通过 `useWorkspaceShortcuts()` 读取浏览器本地映射，键位规范支持 `Ctrl` / `Alt` / `Shift` / `Meta` 组合。
- 新增 `frontend/scripts/shortcut-coverage-check.mjs` 和 `npm run test:shortcuts`：静态扫描所有全局
  `keydown` 入口必须经由 action map，并在 benchmark、run、compare、settings 页面用自定义 keymap
  验证样本切换、图层显隐、几何显隐、视图复位、清除选择和快捷键编辑。
- 设置页预览不启用对象 hover/click hit-test；拖拽时关闭 overlay pointer hit-test，减少鼠标交互开销。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:workspace-settings`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/settings npm run test:settings-preview`
  - `default_role_color_visible = true`
  - `label_color_case_insensitive = true`
  - `label_role_cartesian_product = true`
  - `stroke_width_realtime = true`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:shortcuts`
  - `benchmark_sample_navigation = true`
  - `run_viewer_actions = true`
  - `comparison_viewer_actions = true`
  - `settings_keymap_editor = true`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8765/runs/config_smoke_prompt_params npm run test:viewer-performance`

### 后续防线

- 实例颜色必须按 `label × role` 笛卡尔积表达；禁止再实现成“label 色覆盖 GT/Pred 色”或“状态色覆盖 label 色”的单轴层级。
- 外观页不再暴露全局图层颜色；颜色用户配置只允许落在 `label × role` 单元格上，role 默认色作为固定 fallback。
- 快捷键文档和 UI 不应写死 “G/P/B/L/K” 这类类别列表；新增能力先注册 action，再交给快捷键映射表展示和绑定，并同步扩展 `test:shortcuts` 的静态入口扫描或真实页面行为断言。
- label 匹配 key 可以归一化，但展示文案必须保留原始 label。
- 新增 viewer/settings 配置项时必须先进入 `workspaceSettings.ts` schema，再由 UI 消费；不要在页面里硬写第二份范围。
- 设置页预览类组件需要真实浏览器回归，不能只依赖 React 编译和截图 smoke。

## 2026-05-12: Eval Bench viewer 鼠标缩放和平移卡顿

### 现象

用户反馈 Eval Bench 可视化面板鼠标操作比较卡，主要表现为 run inspector 中滚轮缩放、拖拽平移不够
跟手。这个问题会直接影响对 GT / prediction 叠图的局部排障效率。

### 根因

`CanvasStage` 之前把 `zoom` 和 `pan` 放在 React state 中。每一次 wheel 或 pointermove 都会触发
`setZoom` / `setPan`，从而重渲染整个 `CanvasStage`，连同 SVG overlay、bbox、linestrip、label、
keypoint 和对象 hover 绑定一起重新走 React reconcile。对于实例多或 SVG 元素复杂的样本，这条路径会把
高频鼠标事件变成高频 React 渲染。

### 影响范围

- 影响 Eval Bench benchmark/run/settings/compare 中复用的图像叠图查看器。
- 不影响后端 metric、prediction artifact、job 执行和评测结果。
- 不属于模型能力问题，是前端交互渲染路径过重。

### 修复方式

- `CanvasStage` 的 pan/zoom 改为 ref 持有，并通过 `requestAnimationFrame` 合并后直接更新
  `.image-zoom-layer` 的 CSS transform。
- React state 只保留“是否处于非默认视口”和“是否正在拖拽”这类低频 UI 状态，避免每个 pointermove
  都触发整棵叠图重渲染。
- SVG `InstanceLayer` 改为 memoized component，pan/zoom 期间不重新渲染 GT / prediction overlay。
- wheel delta 归一化并限制单次事件的最大缩放步长；默认滚轮灵敏度降低，避免普通鼠标滚轮出现明显台阶感。
- 缩放/平移活跃期间临时关闭 overlay drop-shadow，并给画布层增加 paint containment，减少浏览器对大 SVG
  filter 的重绘压力。
- 新增 `frontend/scripts/viewer-performance-check.mjs` 和 `npm run test:viewer-performance`，用真实浏览器
  在 run inspector 上连续 wheel + drag，并通过 `perf=1` 调试计数确认高频交互不会重渲染 overlay。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py`
- `EVAL_BENCH_URL=http://127.0.0.1:8765/runs/config_smoke_prompt_params npm run test:viewer-performance`
  - `canvas_renders_during_pan_zoom = 3`
  - `gt_layer_renders_during_pan_zoom = 0`
  - `pred_layer_renders_during_pan_zoom = 0`
- `EVAL_BENCH_URL=http://127.0.0.1:8771/runs/config_smoke_prompt_params INTERACTION_SMOKE=1 npm run render-check`

### 后续防线

- viewer 里的高频交互不能直接落到 React state；优先用 ref + RAF 更新 transform。
- 新增叠图交互能力时，必须跑 `test:viewer-performance`，确认 pan/zoom 不重渲染 heavy overlay layer。
- 只有样本切换、图层开关、label filter、对象 lock/hover 这类语义变化才允许触发 overlay React 渲染。

## 2026-05-11: Eval Bench 页面廉价感、主画布不突出和组件重复

### 现象

用户多次反馈 Eval Bench dashboard 像表单堆叠页面：文字和按钮过多、可视化画布不是主区域、局部
容器显示不全、展开面板容易挤占工作区，整体缺少类似 FiftyOne/CVAT 这类工程工作台的密度和质感。
在真实 run inspector 渲染检查中也发现，主画布宽度只有 586px，而右侧对象检查器有 304px，图像区域
没有成为视觉中心。

### 根因

前端早期迭代把页面功能直接堆在 `main.tsx` 和单个大样式文件里，低频操作表单、空状态、面板标题、
表格、tabs 等重复结构没有统一组件边界。样式层也在基础样式后持续追加局部覆盖，缺少一个明确的
dashboard 设计层，导致信息架构、视觉语言和工作区比例一起漂移。

### 影响范围

- 影响 Eval Bench dashboard 的可用性、审美质量和长期维护成本。
- 影响 benchmark/run/settings/compare 等需要长期盯图排障的页面。
- 不影响 prediction artifact、metric 计算、job 执行语义和模型能力评估结论。

### 修复方式

- 新增 `frontend/src/ui.tsx`，抽出 `DataTable`、`WorkspaceTabs`、`PanelTitle`、`SectionHeader`、
  `EmptyState`、`Badge` 和 `ActionPanel`，减少重复 UI 状态源。
- 新增 `frontend/src/design.css` 作为独立视觉设计层，统一工程工作台色彩、字体、面板、表格、tabs、
  action panel、viewer、settings preview 和 overlay 视觉效果。
- 低频创建/导入/服务登记表单改用统一 `ActionPanel`，默认折叠，避免挤占主工作区。
- 缩窄 benchmark/run 样本栏和 viewer 对象检查器的默认宽度，保留可拖拽调整，使图片画布成为主区域。
- 移除未使用的前端错误通知 helper，避免继续保留临时桥接代码。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- Dashboard render check：
  - `EVAL_BENCH_URL=http://127.0.0.1:8769/ npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8769/benchmarks/multitask_val_v1 INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8769/runs/config_smoke_prompt_params INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8769/settings INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8769/compare npm run render-check`

### 后续防线

- Dashboard 不能把低频管理表单常驻铺满主页面；应优先使用折叠面板、模板化表单和工作区侧栏。
- 任何 inspector / compare / settings 变更都必须保证图片画布是主要区域，并用 render-check 约束
  画布与侧栏比例。
- 新增页面组件时先检查 `ui.tsx` 是否已有通用组件；不要在页面里复制新的 table、empty、badge、
  action panel 或 tabs 实现。
- 视觉层和基础布局层要分开维护；大范围审美调整优先进入 `design.css`，避免在业务组件里写局部补丁。

## 2026-05-11: Eval Bench ephemeral vLLM runtime 成功后残留进程

### 现象

用户创建的 Eval Bench 评测 job 已经结束，但由 job 启动的临时 vLLM 进程没有自动关闭，需要用户手动
kill。表面上 job 状态为 succeeded，但 GPU 和端口仍可能被残留 runtime 占用。

### 根因

`_stop_ephemeral_runtime()` 只在 `Popen` 父进程仍存活时才发送清理信号；如果 vLLM launcher 父进程已经
退出，但同一个 process group 里的 engine/worker 子进程仍然存活，旧逻辑会因为 `process.poll() is not
None` 提前返回，导致子进程残留。Eval Bench 以 `start_new_session=True` 启动 runtime，正确的生命周期边界
应是整个 process group，而不是单个父进程。

### 影响范围

- 影响 Eval Bench ephemeral runtime 模式下的 vLLM job 清理。
- 可能导致 GPU 显存、端口和后台进程残留，进而影响后续 job 调度。
- 不影响 prediction artifact、metric 计算和模型能力评估结论。

### 修复

- `_stop_ephemeral_runtime()` 改为始终向 runtime process group 发送 `SIGTERM`，即使 launcher 父进程已经
  退出也继续清理同组子进程。
- 等待 process group 退出；超时后发送 `SIGKILL` 兜底。
- 如果 `SIGKILL` 后 process group 仍存在，写 warning log，避免静默残留。

### 回归测试

- 覆盖父进程仍存活时的 process group 清理。
- 覆盖父进程已退出但子进程仍存活时的 process group 清理。
- 覆盖 worker 成功路径会关闭 ephemeral runtime。
- 覆盖 worker 异常路径也会关闭 ephemeral runtime。
- 本轮执行：`PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_worker.py`
- 本轮执行：`PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests`

### 后续防线

- 任何由 job 启动的临时外部服务，都必须以 process group/session 为生命周期边界。
- 新增 runtime backend 时必须补成功和失败两条生命周期测试，不能只测任务状态。
- Job succeeded 只代表评测逻辑结束；runtime cleanup 必须作为独立验收条件进入测试。

## 2026-05-11: Eval Bench job 同步执行导致前端状态一直停在排队中

### 现象

用户在 Dashboard 任务中心点击“处理下一条”后，job 实际已经被 worker claim 并开始执行，但前端表格
仍长时间显示“排队中”。页面也没有样本级进度、当前阶段、runtime log tail 或实时监控区域，无法判断
是在启动 vLLM、推理、计算指标，还是已经卡住。

### 根因

`/api/jobs/process-next` 直接同步调用 `EvalBenchWorker.process_next()`，HTTP 请求会一直阻塞到整个
推理和评估完成才返回。虽然数据库中的 `claim_next_job()` 会把状态改为 `running`，但前端 mutation
仍在等待响应，任务列表没有及时重新拉取，所以用户看到的是点击前的 queued 缓存。worker 也只在大阶段
边界写少量 metadata，没有持续写 `done/total/current_sample` 这类可监控进度。

### 影响范围

- 影响 Dashboard 任务中心的状态展示和长任务可观测性。
- 不影响 CLI `process-next-job`；CLI 同步执行仍然合理。
- 不影响 metric 计算和 run artifact 格式。

### 修复

- Dashboard API 的 `/api/jobs/process-next` 改为：先 claim queued job 并立即返回 running job，再用
  后台线程执行 worker。
- CLI 仍保留同步 `EvalBenchWorker.process_next()` 行为，便于脚本和终端使用。
- Worker 新增 `process_job(job_id)` 和统一 `_update_progress()`，在 resolving、starting runtime、
  prepare run、inference、evaluating、succeeded、failed 等阶段持续写 job metadata。
- 推理循环按 sample 更新 `progress_done`、`progress_total`、`progress_current_sample` 和
  `progress_message`。
- Dashboard 新增 `/api/jobs/{job_id}/logs`，读取当前 job 的 `runs/<run_id>/logs/runtime.log` tail。
- 前端任务中心每 2 秒轮询 job record；有 running job 时显示实时监控面板、进度条、当前阶段、当前
  sample 和 runtime log tail；runtime log 每 3 秒刷新。
- Dashboard claim 新 job 前会检查已有 running job 的 `dashboard_worker_pid` 和 `runtime_pid` 是否
  仍存活；旧 dashboard 进程或 vLLM runtime 还在执行时，不会误启动第二个评测 job。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_worker.py projects/eval_bench/tests/test_dashboard.py`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- Dashboard 中会运行较久的 action 不应同步阻塞 HTTP 请求；应尽早返回可轮询的持久化状态。
- 长任务必须持续写 progress metadata，至少包含阶段、done/total、当前样本和更新时间。
- 前端不能只依赖 mutation 返回值刷新长任务状态，必须对 job registry 做短间隔轮询或等价实时订阅。
- 如果新版本 dashboard 与旧版本阻塞进程并存，必须用持久化 pid 检查区分“活着的 running job”和
  “孤儿 running 状态”，避免绕开旧端口时并发启动第二个重型 job。

## 2026-05-11: Eval Bench 需要自动调度队列而不是人工推进

### 现象

Dashboard 的任务中心仍然暴露“处理下一条”式的人工推进入口。用户需要判断何时启动 eval，且同一时间
只能按单个 running job 的思路理解队列；任务运行日志也常驻显示在主队列里，占用了列表空间。

### 根因

队列推进逻辑落在 dashboard action 上，而不是独立的顶层调度器。后端没有一个统一组件同时看 queued
job、live running job、CUDA 设备声明和并发上限，也没有把 runtime log 和队列摘要解耦。

### 影响范围

- 影响 Eval Bench dashboard 的自动化程度和多任务吞吐。
- 不影响已有 CLI `process-next-job` 的同步执行语义。
- 不影响 run artifact、prediction snapshot 和 metric report 格式。

### 修复

- 新增 `EvalBenchOrchestrator`，Dashboard 启动时自动后台运行。
- Orchestrator 按周期扫描 queued eval job，根据 live running job 数、`cuda_visible_devices`、
  ephemeral runtime 端口和 `tensor_parallel_size` 判断是否可调度；CUDA 设备与端口都不冲突的 job
  可并发启动。
- 新增 `EvalBenchDatabase.claim_job(job_id)`，支持跳过资源暂不可用的 queued job 后 claim 其他可运行 job。
- Dashboard 增加 `/api/scheduler/status`，前端任务中心展示自动调度状态、运行数、排队数、并发上限和占用
  CUDA 设备。
- 前端移除主队列中的手动推进入口和常驻 runtime log；点击 job row 后才打开嵌套详情面板读取完整日志。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_orchestrator.py projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- Dashboard 长任务入口应优先进入 queue，由 orchestrator 统一调度，不再依赖用户手动推进。
- 新增 job kind 时要补充资源声明和调度规则，避免每类 job 自己发明启动策略。
- 高频列表只展示摘要状态；完整 runtime log、manifest 和 traceback 应进入按需打开的详情面板。

## 2026-05-11: Eval Bench 服务入口重复与错误可观测性不足

### 现象

Dashboard 中“模型服务”和“任务中心”同时出现服务管理入口，用户难以判断长期 service 与单次 job
runtime 到底应该在哪里管理。评测 job 失败后，前端只看到失败状态，缺少弹窗提醒和 request id；
后端也没有统一的 dashboard 日志文件，只能靠 job runtime log 或终端输出排查。工作台设置里的叠图
预览没有稳定样例图，演示效果不如真实 sample viewer。

### 根因

前端页面职责没有严格对应 Control Plane 对象：Service registry 被重复暴露在 Jobs 页和 Services
页；job failure 只写入 job record，API 层的 HTTP 错误和 worker exception 没有统一落到 store
日志。前端 API client 只抛出异常，没有把错误变成用户可见的 toast。设置页预览虽然复用
`CanvasStage`，但缺少持久化静态样例图和固定 demo instances。

### 影响范围

- 影响 Eval Bench dashboard 的可用性、失败排障和服务/job 生命周期理解。
- 不影响 evaluator、metric 和 prediction artifact。
- 不影响 `src/shaft` 训练主链。

### 修复

- Jobs 页只保留任务队列和新建评测；长期模型服务只在 Services 页管理。
- Store 增加 `eval_bench_store/logs/backend.log`，FastAPI dashboard 启动时配置 `eval_bench`
  logger。
- Dashboard API 增加 request logging middleware 和 HTTPException handler；4xx/5xx 都写入
  backend log，并把 `X-Eval-Bench-Request-Id` 返回给前端。
- 新增 `/api/logs/backend`，用于读取 backend log tail。
- Worker 捕获异常时使用 `LOGGER.exception` 写 traceback，并把可用的 `runtime_log_path` 记录到
  failed job metadata。
- 前端 API client 在 HTTP error 时派发全局错误事件，Shell 中的 `ToastHub` 统一弹出失败原因和
  request id；处理下一条任务后如果 job 失败也立即弹出提醒。
- 工作台设置叠图预览改用 `projects/eval_bench/static/settings_preview.svg` 作为稳定样例图，再叠加
  固定 GT/Pred demo instances，保持和正常 sample viewer 相同的 `CanvasStage` 交互。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py`
- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- `EVAL_BENCH_URL=http://127.0.0.1:8766/settings INTERACTION_SMOKE=1 npm run render-check`

### 后续防线

- 长期共享模型服务只能放在 Service registry；job 专属 vLLM 进程只能作为 job runtime 记录，不能在
  Jobs 页再做一套 service 管理。
- Dashboard API 的新错误路径必须带 request id，并写入 `eval_bench_store/logs/backend.log`。
- 前端所有会触发 API mutation 的动作都应能把失败原因变成用户可见反馈，不能只依赖表格状态变化。
- 工作台设置里的可视化偏好预览必须使用稳定 sample asset 和真实 viewer 组件，避免和实际检查器
  表现分叉。

## 2026-05-11: Eval Bench 默认 vLLM tensor parallel 配置非法导致 job 秒失败

### 现象

在 Dashboard 新建评测任务并开始执行后，job 几秒内变成 `failed`。Job error 为：

```text
runtime process exited before ready: exitcode=1
```

对应 `runs/<run_id>/logs/runtime.log` 中 vLLM 启动失败：

```text
Total number of attention heads (32) must be divisible by tensor parallel size (3).
```

### 根因

Eval Bench 的默认 `eval_job` 模板把 `tensor-parallel-size` 写成了 `3`，但当前 Qwen3VL text
config 的 `num_attention_heads=32`。vLLM tensor parallel 要求 attention heads 能被 TP size 整除，
所以 3 卡张量并行对该模型非法。此前 preflight 只检查模型路径、benchmark、task、prompt 和端口，
没有读取模型 config 校验 vLLM TP 约束。

### 影响范围

- 影响从 Dashboard 默认模板创建的 ephemeral vLLM eval job。
- 不影响已经登记的长期 service，也不表示模型权重或 benchmark 有问题。
- 这是 runtime 参数配置问题，不是 metric、parser 或模型能力问题。

### 修复

- 默认 `eval_job` 模板改为单卡保守值 `CUDA_VISIBLE_DEVICES=0`、`tensor-parallel-size=1`。
- 默认 runtime env 增加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，降低混合 GPU 顺序导致的误解。
- Worker 启动 ephemeral runtime 时透传 manifest 中的 runtime env，而不是只透传 `CUDA_VISIBLE_DEVICES`。
- Preflight 读取 `model_path/config.json` 中的 `text_config.num_attention_heads`，提前拒绝不能整除
  attention heads 的 `tensor_parallel_size`。
- Preflight 同时检查 `CUDA_VISIBLE_DEVICES` 数量不能少于 `tensor_parallel_size`。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_worker.py`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 新增或修改 vLLM job template 时，不能只看 GPU 数量；必须确认 TP size 是模型 attention heads 的因子。
- Dashboard preflight 必须覆盖会导致 runtime 秒退的静态配置错误，避免用户启动 job 后才从 runtime log 查原因。
- 允许用户在 manifest 中自由改 GPU 和 TP，但错误组合必须在 preflight 被明确报出。

## 2026-04-28: online eval 左 padding completion 切片污染 keypoint JSON 解析

### 现象

`grounding_row_bucket` 训练第一次在线 eval 中，`keypoint_arrow` 指标明显异常：

- `parse_success=0.5125`
- `keypoint_pck=0.3881`

从任务难度和已有数据质量看，keypoint 不应该弱到这个程度，尤其 parse success 不应该只有一半。

### 根因

在线 eval 的 prompt collator 使用 `left padding`，但 decoder-only 生成结果切 completion 时使用了每条样本的 `attention_mask.sum()`：

```python
completion_ids = row[prompt_length:]
```

对于左 padding batch，HF decoder-only `generate()` 返回的是：

```text
[padded_input_ids, generated_completion_ids]
```

completion 的起点应该是 batch padded input width，也就是 `input_ids.shape[1]`，不是每条样本自己的非 pad token 数。短样本用 `attention_mask.sum()` 会切早，把 prompt 尾部一起解码成 prediction。

keypoint prompt 里包含枚举列表：

```text
["solid", "dashed"]
["straight", "rounded", "curved"]
```

当 prompt 尾部混进 prediction 后，`json_object` codec 会先看到 `[`，解析成 JSON list，再因为期望 object 报 `json_type_error`。这会把本来可能合法的模型输出记为 parse failure。

### 影响范围

- 主要影响 generation-based online eval。
- 只要满足以下条件就有风险：
  - decoder-only 模型
  - `left padding`
  - 按 `attention_mask.sum()` 切 generated row
  - prompt 尾部含 JSON-like 片段，且 prediction codec 对 JSON 顶层类型敏感
- 对 `keypoint_arrow` 影响尤其大，因为 prompt 中有 JSON-style 枚举列表。

### 修复

- `ShaftOnlineEvalRunner` 对 decoder-only 输出统一按 `prepared["input_ids"].shape[1]` 切 completion。
- encoder-decoder 模型仍按生成输出本身解码，不追加 input prefix 假设。
- 新增回归测试覆盖左 padding 下 prompt 尾部包含 list、prediction codec 要求 `json_object` 的场景。

### 同步发现的 metric 标准问题

`keypoint_arrow` 的 `keypoints_2d` 使用 0-1000 bin 坐标，但 `keypoint_pck` 曾用图片宽高作为 5% 容差尺度。对于几十像素的小 crop，容差会被压到几格，明显偏严。

修复为：

- `keypoint_pck` 默认使用 `normalized_1000` 坐标尺度。
- 配置中显式写入：

```yaml
- name: keypoint_pck
  params:
    coordinate_space: normalized_1000
    num_bins: 1000
```

如未来评估像素坐标 keypoint，需要显式设置 `coordinate_space: image`。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_slices_left_padded_decoder_prompts_at_input_width`
- `tests/test_online_eval.py::test_keypoint_pck_uses_normalized_coordinate_scale_by_default`

本次验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_online_eval.py
.venv/bin/ruff check src/shaft/training/online_eval.py src/shaft/metrics/builtin.py tests/test_online_eval.py
.venv/bin/python -m compileall src/shaft/training/online_eval.py src/shaft/metrics/builtin.py tests/test_online_eval.py
```

### 后续防线

- 所有 generation eval / infer 路径都要明确 completion slice invariant：
  - decoder-only: completion starts at padded input width
  - encoder-decoder: decode generated sequence directly
- 不允许在 left padding generation 路径用 `attention_mask.sum()` 作为 completion 起点。
- 新增结构化任务 prompt 时，如果 prompt 含 JSON 示例、枚举列表或 schema 片段，必须额外检查 codec 是否可能被 prompt 泄漏污染。
- 指标中坐标尺度必须显式化：`bbox_2d` / `keypoints_2d` 如果是 0-1000 bin，metric 不得默认退回图片像素尺度。

## 2026-04-29: 单进程多卡触发 DataParallel 破坏 Qwen3VL 视觉张量对齐

### 现象

从 `checkpoint-23640` resume 训练时，第一步 forward 在 Qwen3VL visual tower 中报错：

```text
RuntimeError: The size of tensor a (1106) must match the size of tensor b (1676) at non-singleton dimension 0
```

调用栈中出现：

```text
torch.nn.parallel.data_parallel.py
```

这说明当前训练不是 DDP，而是单进程可见多张 CUDA 卡后被 Hugging Face Trainer 包成了 PyTorch `DataParallel`。

### 根因

Qwen3VL 的多模态 batch 中：

- `pixel_values` 是所有图片 patch 拼接后的变长张量，第 0 维是 patch 数。
- `image_grid_thw` 是按图片计数的网格元数据，第 0 维是图片数。

PyTorch `DataParallel` 会按第 0 维独立切分每个 tensor。它不知道 `pixel_values` 与 `image_grid_thw` 之间的语义对应关系，于是会把 patch 张量和 grid 元数据切到不一致的 shard。进入 visual tower 后，patch embedding 的长度与根据 `image_grid_thw` 生成的位置 embedding 长度不一致，最终在：

```python
hidden_states = hidden_states + pos_embeds
```

处报维度不匹配。

### 影响范围

- 影响所有 Qwen3VL 类 decoder-only 多模态训练路径，只要满足：
  - 单进程启动
  - 多张 CUDA 卡对进程可见
  - 没有用 `torchrun` / DDP
- 与 checkpoint 本身无关，也不是 `lm_head.weight` missing warning 的直接原因。
- `per_device_train_batch_size` warning 不是这次维度错误的根因；真正触发点是 `DataParallel` 对多模态变长视觉张量的错误切分。

### 修复

新增训练 topology guard：

- 当 CUDA 可用、可见 GPU 数量大于 1、且没有分布式启动环境变量时，训练启动阶段直接报错。
- 报错信息明确提示：
  - 单卡：使用 `CUDA_VISIBLE_DEVICES=<id> python scripts/train.py ...`
  - 多卡：使用 `torchrun` / DDP
- guard 放在模型加载前，避免先加载大模型再在第一步训练时炸。

### 回归测试

- `tests/test_pipeline_sft.py::test_training_topology_rejects_single_process_data_parallel`
- `tests/test_pipeline_sft.py::test_training_topology_allows_distributed_launch`

### 后续防线

- 多模态训练不允许依赖 PyTorch `DataParallel`。
- 任何训练入口只要可能看到多张 CUDA 卡，都必须显式区分：
  - 单卡单进程
  - DDP 多进程
  - 非法的单进程多卡
- 如果未来新增模型族，其视觉输入中存在按 patch 展平、按图片记录 metadata 的结构，也必须继承这条 topology 约束。

## 2026-04-29: DDP online eval 显示口径和样本去重必须与单卡一致

### 现象

cuda1/cuda2 的 DDP smoke 训练和 online eval 能跑通，但 progress bar 显示为：

```text
online_eval 1/1 batch
```

同一份 val 在单卡上显示为：

```text
online_eval 2/2 batch
```

这说明 DDP 下显示的是 rank0 本地 dataloader 进度，而不是全局 eval 进度。进一步检查发现，online eval 会 all-gather 各 rank 预测再聚合 metric，但没有对 DistributedSampler padding 可能带来的重复样本去重。

### 根因

- 显示层：progress bar 只在 rank0 创建，total 取 rank0 本地 dataloader 的 batch 数。
- metric 层：DDP all-gather 后直接聚合全部 entries。如果 eval 样本数不能被 world size 整除，分布式 sampler 可能 padding 重复样本，重复项会进入平均指标。

### 影响范围

- DDP online eval 的最终 metric 主路径是全局 all-gather 聚合，方向正确。
- 当样本数能被 world size 整除时，metric 与单卡一致。
- 当样本数不能被 world size 整除时，若 sampler padding 重复样本，metric 可能被重复样本轻微影响。
- progress bar 在 DDP 下不是单卡同口径，会低估全局 eval 总量。

### 修复

- `ShaftOnlineEvalRunner.aggregate_samples()` 聚合前按 `(dataset_name, sample_id, image_path)` 去重。
- DDP progress bar 改为全局 sample 口径：
  - total 使用 dataloader dataset 的全局长度。
  - rank0 每个 batch 按 `local_batch_size * world_size` 更新，并 cap 到 total，避免 padding batch 超出总量。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_deduplicates_gathered_samples_before_metrics`

### 后续防线

- DDP eval 的 metric 聚合必须以全局唯一样本为准，不得让 sampler padding 改变指标。
- DDP eval 的显示口径必须明确是全局样本进度，或在文案中显式标注为 local rank 进度。

## 2026-04-30: GRPO/vLLM 绕过 SFT collator 导致图像 token 预算失效

### 现象

在 cuda1 上尝试单卡 `vllm.mode=colocate` 的 GRPO smoke 时，vLLM 能加载并完成 CUDA graph 初始化，但第一步 rollout 在输入校验阶段失败：

```text
The decoder prompt (length 12324) is longer than the maximum model length of 8192.
```

此前 `vllm.max_model_length=4096` 时也出现过同类错误，某个样本的 prompt 长度已经达到 6205 tokens。

### 根因

SFT/DPO/PPO collator 会通过 `model_adapter.build_processor_inputs(..., min_pixels, max_pixels)` 把 `data.max_pixels` 传给 processor。

GRPO 使用 TRL `GRPOTrainer`，不走 Shaft 的 SFT collator。`GRPODataset` 之前直接返回原始 PIL 图像，TRL/vLLM 会按自己的 VLM 路径处理图像，导致 `data.max_pixels=262144` 没有生效。高分辨率图像被展开成过多 multimodal tokens，最终超过 vLLM context。

这不是模型能力问题，也不是 reward/metric 问题，而是 GRPO 数据适配层没有继承 Shaft 图像 token 预算语义。

### 影响范围

- 影响 VLM GRPO，尤其是 `use_vllm=true` 的 rollout。
- 非 vLLM GRPO 也会受到影响，因为 TRL 的 VLM prompt/forward 处理同样绕过 Shaft collator。
- SFT 主链不受影响，SFT collator 已显式传入 `min_pixels / max_pixels`。

### 修复

- `GRPODataset` 新增 `min_pixels / max_pixels` 参数。
- `ShaftRLHFPipeline` 构建 GRPO dataset 时传入 `config.data.min_pixels / max_pixels`。
- `GRPODataset` 在样本进入 TRL 前按像素预算调整 PIL 图像，避免原始大图撑爆 multimodal token 数。
- GRPO 配置结构同步改为：
  - `rlhf.grpo.rollout`
  - `rlhf.grpo.vllm`
  并保留旧 flat 字段作为兼容入口。

### 回归测试

- `tests/test_pipeline_rlhf.py::test_grpo_dataset_applies_image_pixel_budget`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_uses_sft_dataset_for_grpo`
- `tests/test_config_loader.py::test_load_config_supports_grpo_reward_config`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 新增 RLHF/VLM 路径时，必须确认是否经过 Shaft collator；如果不经过，图像 token 预算要在 dataset adapter 或算法 adapter 层显式落地。
- 不能只调大 `vllm.max_model_length` 来掩盖图像预算失效；必须先确认 `data.max_pixels` 对实际 rollout prompt 生效。
- `rollout.max_completion_length` 只限制生成长度，不能替代 prompt multimodal token 控制。

## 2026-04-30: GRPO/vLLM colocate sleep mode 触发每步磁盘重载 checkpoint

### 现象

启动 `grounding_grpo_vllm_colocate_g8_bs32_1024` 后，训练能正常进入 step，但每个 train step 前都会反复出现：

```text
Loading safetensors checkpoint shards: 0/2
Loading safetensors checkpoint shards: 2/2
```

这和预期不一致。GRPO 中 vLLM rollout 副本确实需要随 policy 动态更新，但正常应从训练进程内存中的当前参数同步到 vLLM，而不是每步从磁盘 checkpoint 重新加载 safetensors。

### 根因

配置中开启了：

```yaml
rlhf:
  grpo:
    vllm:
      enable_sleep_mode: true
```

当前 TRL/vLLM colocate 路径中：

- `sync_weights()` 会把训练中的 policy 参数同步到 vLLM 副本。
- `generate()` 在 `enable_sleep_mode=true` 时会唤醒 vLLM，并调用 `collective_rpc("reload_weights")`。
- 在当前 `vLLM 0.19.0` 环境下，这个 `reload_weights` 会触发从磁盘 checkpoint shard 重新加载权重。

因此日志中每个 step 的 safetensors reload 不是正常的 policy 内存同步，而是 sleep/wake 机制引入的额外磁盘重载。

### 影响范围

- 影响 GRPO `vllm.mode=colocate` 且 `enable_sleep_mode=true` 的训练。
- 性能上会显著拖慢 step，因为每步多了一次 checkpoint shard 读取和加载。
- 语义上存在风险：如果 `reload_weights` 从初始 checkpoint 重载，可能覆盖刚通过 `sync_weights()` 同步到 vLLM 的当前 policy 权重，使 rollout 退回旧权重。
- `enable_sleep_mode=false` 时，vLLM 副本常驻显存，不触发这类 sleep/wake reload 路径。

### 修复

- 将 `configs/train/train_grpo_4b_grounding.yaml` 中的：

```yaml
enable_sleep_mode: true
```

改为：

```yaml
enable_sleep_mode: false
```

关闭后，vLLM 推理副本常驻显存。只要不 OOM，就优先使用这一设置，保证 rollout 权重同步语义和训练速度都更稳定。

### 回归测试

本问题主要通过训练日志验证：

- 正常现象：vLLM 初始化阶段加载 checkpoint。
- 异常现象：每个 train step 都出现 `Loading safetensors checkpoint shards`。
- 修复后应重新启动同一训练命令，确认 step 间不再反复磁盘加载 safetensors。

### 后续防线

- GRPO/vLLM 的权重同步必须区分两种语义：
  - 正确：optimizer step 后从训练进程当前参数同步到 vLLM。
  - 错误：每步从磁盘 checkpoint 重新加载 vLLM 权重。
- 开启 vLLM `sleep mode` 前必须先做多 step canary，确认不会反复触发 safetensors reload。
- 如果关闭 sleep mode 后 OOM，优先考虑降低 `gpu_memory_utilization`、`max_model_length`、`max_completion_length`，或改用独立 vLLM server/单独 GPU rollout，而不是接受每步磁盘重载。

## 2026-04-30: GRPO reward wrapper 导致 W&B per-reward 指标不可读

### 现象

检查 GRPO 监控项时发现，多个 reward function 传给 TRL 后函数名都叫 `_reward_func`。TRL 使用 `reward_func.__name__` 作为 W&B metric key，因此多个 reward 会写入同一类指标：

```text
rewards/_reward_func/mean
rewards/_reward_func/std
```

同时，reward weight 被提前乘在 wrapper 返回值里，导致 per-reward mean/std 是加权后的数值。例如 `parse_success` 权重为 `0.05` 时，W&B 中该项最高只能到 `0.05`，不能直接看作 parse success rate。

### 根因

`build_grpo_reward_functions()` 为每个 reward 创建闭包，但没有设置可区分的 `__name__`。并且 reward 权重被内联进闭包返回值，而不是交给 TRL 原生的 `reward_weights`。

### 影响范围

- 影响 GRPO W&B 监控可读性。
- 不影响总 reward 的数学结果，但会让 per-reward 监控误导：
  - 无法区分 `parse_success` 和 `grounding_iou`
  - 无法从 per-reward mean 直接读出原始 parse rate / IoU reward

### 修复

- 每个 GRPO reward wrapper 设置稳定名称：
  - `grpo_reward_parse_success`
  - `grpo_reward_grounding_iou`
- reward function 返回原始 reward。
- `build_trl_grpo_config()` 将配置中的权重传给 TRL `reward_weights`，由 TRL 聚合总 reward。

### 回归测试

- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_exact_match_and_parse_success`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_iou`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 新增 reward function 时，W&B key 必须稳定且可区分。
- per-reward 指标应记录原始 reward；权重应放在聚合层，避免监控值被缩放后难以解释。

## 2026-04-30: DDP 训练时 Shaft summary 元数据并发写入失败

### 现象

使用两卡启动 GRPO/vLLM colocate 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train.py rlhf ...
```

vLLM 初始化成功，但训练创建 optimizer 时 rank0 报错：

```text
FileNotFoundError: shaft_optimizer_summary.tmp -> shaft_optimizer_summary.json
```

### 根因

`ShaftOptimizerMixin.create_optimizer()` 在每个 DDP rank 上都会调用
`write_resolved_optimizer_summary()`。该函数使用固定临时路径
`shaft_optimizer_summary.tmp` 后再 `replace()` 到正式 json。多个 rank 同时写同一个
tmp 文件时会产生竞争：一个 rank 已经把 tmp replace 掉，另一个 rank 再 replace
同一路径时就会找不到文件。

同类风险也存在于 `shaft_finetune_summary.json` 写入。
训练结束阶段的 `ensure_hf_export_layout()` 和 `prune_root_output_layout()` 也是
Shaft 自己的 run 级文件操作，不能在所有 rank 上重复执行。

这不是模型能力问题，也不是 vLLM rollout 问题，而是训练元数据落盘没有遵守
DDP single-writer 语义。

### 影响范围

- 影响所有 DDP 训练路径，包括 SFT 和 RLHF。
- 单卡训练不受影响。
- 小规模 DDP smoke 可能不稳定复现，因为 rank 间时序足够错开时不会撞到同一个 tmp 文件。

### 修复

- `shaft_optimizer_summary.json` 只在 rank0 写入和记录启动日志。
- `shaft_finetune_summary.json` 只在 rank0 写入和记录启动日志。
- final export layout 校验与 root output prune 只在 rank0 执行。
- 非 rank0 仍正常创建 optimizer 和训练，只跳过 run 级 summary 落盘。

### 回归测试

- `tests/test_training_modules.py::test_optimizer_summary_is_written_only_on_rank_zero`
- `tests/test_pipeline_sft.py::test_run_sft_rank_nonzero_skips_run_level_file_ops`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_rank_nonzero_skips_run_level_file_ops`
- `tests/test_smoke_distributed.py::test_torchrun_train_eval_smoke`

### 后续防线

- DDP 下 run 级元数据必须是 single-writer，优先 rank0 写入。
- 如果未来确实需要多 rank 分别写文件，文件名必须包含 rank 或使用独立子目录，不能共享固定 tmp 路径。
- 多卡 smoke 不应只验证 forward/eval，也要覆盖 optimizer 创建和 run-level metadata 写入路径。

## 2026-04-30: GRPO reward 误把 partial JSON 当作完整有效输出

### 现象

检查 grounding GRPO reward 设计时发现，模型如果只输出一个未闭合 JSON 起始符：

```text
[
```

`json_list` codec 会将其修复为 partial `[]`。在空目标 hard negative 样本上，
该输出可以同时拿到 `parse_success=1.0` 和 `grounding_iou=1.0`，形成格式层面的
reward hacking 空间。

### 根因

GRPO reward 使用 codec 的 `decoded.valid` 判断解析成功，但没有区分
`decoded.partial`。而 JSON codec 的 lenient repair 是为诊断和容错 eval 服务的，
不应在训练 reward 中等价为完整正确输出。

### 影响范围

- 影响 GRPO 中使用 JSON codec 的 reward：
  - `parse_success`
  - `exact_match`
  - `grounding_iou`
- 对 grounding hard negative 样本尤其敏感，因为空目标 `[]` 是合法答案。
- 这是 reward 语义偏差，不是模型能力问题，也不是 eval metric 的误判。

### 修复

- GRPO reward 只接受 `decoded.valid and not decoded.partial` 的完整解析结果。
- partial decode 在 `parse_success` 中计为 0。
- partial decode 在 `exact_match` 和 `grounding_iou` 中直接计为 0，避免修复后的
  `[]`、`{}` 与目标偶然匹配。

### 回归测试

- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_exact_match_and_parse_success`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_iou`

### 后续防线

- 训练 reward 应比离线诊断 codec 更严格；partial repair 可以用于观测，
  不应默认给正 reward。
- Grounding GRPO 监控需要单独记录 `parse_partial_rate`、`pred_empty_rate`、
  `target_empty_rate`、`positive_pred_empty_rate`，避免只看总 reward。
- 空目标 hard negative 应按 bucket 单独监控，防止模型通过过度输出空数组提高局部 reward。

## 2026-05-01: DDP 正常结束后仍打印 process group 清理 warning

### 现象

两卡 GRPO 训练完整跑到 `global_step=1526`，最终 checkpoint 与 `best/` 均正常保存，
但退出阶段打印：

```text
barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
WARNING: destroy_process_group() was not called before program exit
```

### 根因

Shaft 在训练结束阶段会调用 distributed barrier，但没有为 NCCL barrier 显式传入当前
CUDA device id。训练 CLI 退出时也没有显式调用 `torch.distributed.destroy_process_group()`，
因此 PyTorch 在进程退出阶段提示 process group 未主动销毁。

### 影响范围

- 影响 torchrun/DDP 启动的训练退出日志可读性。
- 不影响本次训练结果；本次运行已经成功完成并保存最终模型。
- 该 warning 本身不是 NCCL 通信失败。真正通信失败通常会伴随 timeout、rank 非零退出、
  `ChildFailedError` 或 barrier hang。

### 修复

- `barrier_if_distributed()` 在 NCCL backend 下显式传入当前 CUDA device id。
- 新增 `destroy_process_group_if_initialized()`，仅在 distributed 已初始化时执行销毁。
- 训练 CLI 在 `finally` 中调用销毁 helper，确保成功和异常退出都会清理 process group。

### 回归测试

- `tests/test_distributed_runtime.py::test_barrier_if_distributed_noop_without_dist`
- `tests/test_distributed_runtime.py::test_barrier_if_distributed_passes_nccl_device_ids`
- `tests/test_distributed_runtime.py::test_destroy_process_group_if_initialized_calls_dist_destroy`

### 后续防线

- 新增分布式收尾逻辑必须走 `shaft.utils.distributed`，避免各 pipeline 自己直接操作
  `torch.distributed`。
- torchrun/DDP smoke 除了检查训练完成，也应关注退出阶段是否还有 NCCL/process group 清理 warning。

## 2026-05-03: Grounding GRPO 离线最优点早于最终 checkpoint

### 现象

两卡 grounding GRPO 训练完成后，对 SFT baseline 与 GRPO checkpoint 做离线 grounding eval。
结果显示 `checkpoint-1200` 的 grounding macro F1 略高于 SFT，但最终 `checkpoint-1526`
和 `best/` 回落，且 `best/` 与 final 指标一致。

### 根因

当前 GRPO 配置未开启 online eval，`load_best_model_at_end=false`，因此 `best/` 不是按
validation `det_f1` 选择的 best，而是训练结束后的 final save。同时旧 reward 主要由
`grounding_iou` 驱动，和 offline `det_f1` 并不完全一致；`beta=0.0` 也缺少对 SFT policy
的 KL 约束，容易在 SFT 已较强时出现后段过优化。

这是训练选择与 reward 对齐问题，不是 offline eval/codec/metric 的误判。

### 影响范围

- 影响 grounding GRPO checkpoint 选择；不能默认使用最终 `best/`。
- 影响 reward 优化方向；单看 IoU reward 可能掩盖漏检、重复框和 precision/recall 退化。
- 影响训练监控；没有 online eval 时无法及时发现真实 validation F1 的回落。

### 修复

- GRPO pipeline 支持 `eval.online_metrics_enabled=true`，复用 SFT 的
  `ShaftOnlineEvalRunner` 做 generation-based validation。
- GRPO 配置开启 grounding online eval，使用与 SFT 相同的 grounding eval policy 与
  `max_new_tokens=2048`。
- 新增 `grounding_det_f1` reward，使训练 reward 直接对齐 offline `det_f1` 的
  IoU-threshold matching 语义。
- 新一版 GRPO 配置降低学习率、增加 `beta=0.02`，并把 `grounding_iou` 降为辅助 reward。

### 回归测试

- `tests/test_config_loader.py::test_load_config_supports_grpo_online_eval_dataset_policies`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_det_f1`

### 后续防线

- GRPO 长训必须开启 online eval，并让 `save_steps` 与 `eval_steps` 对齐。
- grounding reward 应至少包含 parse 完整性、det F1 和 IoU 辅助项，避免只优化局部框 IoU。
- 若 `best/` 与 final 指标一致，需要确认是否真的启用了 `load_best_model_at_end` 和
  `metric_for_best_model=eval_final_score`。

## 2026-05-03: Grounding GRPO v2 首步长时间无进度

### 现象

启动两卡 GRPO v2 online eval 训练后，日志停在 `train: 1/1526` 附近。诊断时 rank0
在 GPU0 上持续 100% 利用率，rank1 占用 GPU1 约 85GB 但 GPU 利用率为 0%，输出目录没有
checkpoint 写入。

### 根因

本轮把 `rlhf.grpo.beta` 从 0 调到 0.02 后，TRL GRPO 会创建 reference model 用于 KL。
当前 TRL 的 `create_model_from_path()` 在未显式传 `dtype` 时默认按 `float32` 加载 reference
model；这与主模型 bf16 训练精度不一致，导致显存占用接近卡容量并明显拖慢首步。

同时训练 rollout 的 `max_completion_length=2048` 会让每个 generation batch 采样更长 completion。
这应与 online eval 的 `eval.max_new_tokens=2048` 区分；eval 需要和 SFT 对齐，训练 rollout
不应默认使用同样长的上限。

### 影响范围

- 影响 `beta > 0` 的 GRPO full-finetune 训练，尤其是 colocate vLLM 场景。
- 不影响 `beta=0` 的旧 run，因为旧配置不会创建 reference model。
- 不影响 SFT online eval；这是 GRPO ref model 初始化精度与 rollout 长度组合导致的训练性能问题。

### 修复

- GRPO 的 TRL config 装配会根据 `TrainingArguments` 精度设置 `model_init_kwargs.dtype`：
  - `bf16=true` 时传 `dtype=bfloat16`
  - `fp16=true` 时传 `dtype=float16`
- 训练 rollout 的 `max_completion_length` 调回 1024。
- 保留 online eval 的 `eval.max_new_tokens=2048`，继续与 SFT 的生成评估口径对齐。

### 回归测试

- `tests/test_training_modules.py::test_build_trl_grpo_config_sets_bf16_model_init_kwargs`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 以后只要 GRPO 开启 `beta > 0`，启动日志里 reference model 不应再出现
  `default dtype torch.float32`。
- 训练 rollout 长度和 eval 生成长度必须分开审查；为了监控能力可以让 eval 更长，但训练采样
  应先按吞吐和 reward 可用性选上限。

## 2026-05-03: GRPO online eval 在 step 200 触发 rollout prepare 报错

### 现象

GRPO v2 训练推进到 step 200 后触发 online eval，两个 rank 同时报错：
`TypeError: string indices must be integers, not 'str'`。堆栈显示
`ShaftOnlineEvalRunner.collect_samples()` 调用 `trainer._prepare_inputs(batch)` 后进入了
TRL `GRPOTrainer._prepare_inputs()`，该函数继续调用 `_generate_and_score_completions()` 并按
`x["prompt"]` 读取输入。

### 根因

SFT online eval 里 `trainer._prepare_inputs()` 只做标准 HF batch 设备搬运；但 GRPOTrainer
覆写了同名方法，把它变成训练 rollout 的 generation/scoring 入口，要求输入是 GRPO 样本列表。
online eval 使用的是 `SFTCollator` 产出的模型输入 dict，两者语义不兼容。

这是 online eval 与 GRPO trainer 方法名复用导致的 trainer 接口误用，不是数据、codec 或 metric
本身的问题。

### 影响范围

- 影响 `eval.online_metrics_enabled=true` 的 GRPO 训练。
- 不影响 SFT online eval。
- 不影响 GRPO step 200 前的训练；报错发生在 `_maybe_log_save_evaluate()` 的 eval 阶段。

### 修复

- `ShaftOnlineEvalRunner` 优先调用 trainer 的 `prepare_online_eval_inputs()` hook。
- `ShaftGRPOTrainer.prepare_online_eval_inputs()` 显式调用 HF `Trainer._prepare_inputs()`，只做标准
  batch 准备，绕开 TRL GRPO 的 rollout `_prepare_inputs()`。
- 没有该 hook 的 trainer 继续走原有 `_prepare_inputs()`，保持 SFT 行为不变。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_uses_online_prepare_hook`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets`

### 后续防线

- online eval runner 不应直接假设所有 trainer 的 `_prepare_inputs()` 都是 HF 原始语义。
- 接入新 trainer 时，如果它覆写了 `_prepare_inputs()`、`prediction_step()` 或 dataloader 行为，
  必须显式确认 online eval 的 batch 准备路径不会触发训练专用逻辑。

## 2026-05-05: raw arrow/layout 标注语义归一与噪声清理

本次把 `data/raw_arrow/json` 统一为 `label=arrow + bbox + linestrip` schema；旧
`c0-c7` bbox 标签的单/双头、直/曲、实/虚信息进入 `subattr`，新增 connector 数据缺少
单/双头标注，因此保持 `arrow_type=unknown`。`data/raw_layout/json` 删除了 41 个零宽或零高
噪声实例，并复查同图内同 label、同 bbox 重复数为 0。

两个 raw 数据目录的当前状态和后续注意事项以各自 README 为维护入口：
`data/raw_arrow/README.md`、`data/raw_layout/README.md`。

## 2026-05-06: raw_layout/raw_arrow 合并为统一 raw_data 真源

### 现象

原始 layout 和 arrow 标注分散在两套 raw 目录中，同一张图片可能有 layout 层、arrow 层或只存在
未标注库存图片。继续维护两套 raw 目录会让 split、preview、补标状态和后续派生数据生成出现多处
状态源。

### 根因

`raw_layout` 和 `raw_arrow` 最初服务不同任务，目录结构和 split 独立；但后续多任务训练与补标流程
需要以图片为中心管理 layer 覆盖状态。任务级 raw 目录不能表达“该图只标了 arrow、layout 未标”或
“该图暂未进入任何标注层”的统一状态。

### 影响范围

- raw 真源切换为 `data/raw_data`；旧 `raw_layout` / `raw_arrow` 不再作为新维护入口。
- 训练派生数据仍按任务读取：arrow 使用 arrow layer，layout 使用 layout layer。
- `data/` 被 Git 忽略，实际 raw 数据通过共享目录同步；仓库中维护的是数据管理规则和生成代码。

### 修复

- 合并已有 JSON 标注为 instance-centric `shaft.raw_data.v1`：
  - `annotation.layers` 记录覆盖层，固定按 `layout`、`arrow` 顺序。
  - `annotation.status` 记录每个 layer 的流程状态。
  - layout instance 保持 `label + bbox + extra`。
  - arrow instance 保持 `label + bbox + linestrip + subattr + extra`。
- 将 layout image-only 库存也写入 `raw_data`，使用
  `annotation.layers=[]`、`annotation.status={}`、`instances=[]`，作为未来补标库存，
  不作为任何任务负样本。
- 按任务生成 split：`arrow_train/val` 和 `layout_train/val`；当前只从已标注 layer 中划分。
- Preview 改为按 label 生成：`icon`、`image`、`shape`、`arrow`，并复用
  `shaft.metrics.visualization` 的统一绘制风格。
- `shaft-data-manager` skill 收口为统一 raw_data 维护入口，原数据增强 skill 并入该 skill。

### 回归测试

- 校验 `raw_data/json` 与 `raw_data/images` 一一对应。
- 校验旧 layout/arrow instances 与合并后对应 layer 的 instances 完全一致。
- 校验 split 中每个 stem 都存在 JSON 和图片。
- 校验 preview 数量与每个 label 出现的 JSON 数一致。

### 后续防线

- 不再从旧 `raw_layout` / `raw_arrow` 目录启动新数据维护任务。
- 缺失 layer 不能当作负样本；只有 completed layer 且无 instance 才能表示人工确认负样本。
- raw split 和 preview 只是辅助状态，不替代 `annotation.layers` / `annotation.status`。

## 2026-05-06: SFT arrow/layout/keypoint v2 训练 step 547 OOM 引发 NCCL timeout

### 现象

`train_sft_4b_grounding.yaml` 从旧 SFT best 初始化后，2 卡 full SFT 在 step 547 附近失败。
rank1 在 `cross_entropy` 分配约 10.94 GiB 时 CUDA OOM；rank0 随后卡在 allreduce，30 分钟后
NCCL watchdog 报 `WorkNCCL(ALLREDUCE) timeout` 并终止进程组。

### 根因

首要根因是单卡 micro-batch 过大：`per_device_train_batch_size=4` 配合 4B full finetune、
`max_pixels=1048576` 和长 target 时，loss/logits 计算峰值显存超过 83GB 卡余量。NCCL timeout
是某个 rank OOM 后其它 rank 继续等待 collective 的连带结果，不是通信链路先异常。

日志里的 `Num examples = 22,196` 是当前 sharded mixed sampler 的单 rank 长度；全局 mix 仍按
catalog 权重生成约 44,392 条/epoch，不是数据源只剩 keypoint。

### 影响范围

- 影响当前 4B full SFT v2 配置下的训练稳定性。
- 不表示旧 SFT best checkpoint 损坏。
- 不表示 DDP/NCCL 本身不可用；OOM 后的 allreduce timeout 是预期连锁失败。

### 修复

- 将训练 micro-batch 从 `per_device_train_batch_size=4` 降到 `2`。
- 将 `gradient_accumulation_steps` 从 `2` 提到 `4`，保持 global batch size 仍为 16。
- 将 online eval batch size 从 `8` 降到 `2`，避免 epoch 2 生成式 eval 再次触发显存峰值。
- 建议重跑时设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`，降低显存碎片导致的边界 OOM 风险。

### 回归测试

- 配置加载检查：确认 batch/accum/eval batch 字段解析为新值。
- 训练重跑时以前 600 step 为 canary，若通过原失败点，说明本轮 OOM 风险已降低。

### 后续防线

- full finetune + multimodal large pixel budget 不应默认使用 per-device train batch 4。
- 看到 NCCL timeout 时先检查其它 rank 是否 OOM、异常退出或数据读取失败，再判断通信问题。
- 训练日志里的 `Num examples` 在 sharded custom sampler 场景下可能是单 rank 视角，数据 mix
  应用 sampler/global quota 复核。

## 2026-05-10: Eval Bench job/service/runtime 生命周期重新分层

### 现象

Eval Bench 初版同时存在 job queue、service registry 和前端创建表单，但 job 参数仍偏固定表单，
一次性 vLLM runtime 与长期 service 的边界不够清楚。继续扩展时，容易把“开一个 eval job 就启动
一套临时模型后端”和“长期共享 vLLM endpoint”混成同一类状态，导致任务记录、服务记录和 run
manifest 之间缺少可复盘的配置真源。

### 根因

Control Plane、Execution Plane、Artifact Plane 没有显式分层；前端 job 创建直接暴露固定字段，
后端 worker 只消费扁平 payload。这样短期能跑通，但无法承载可编辑参数模板、preflight、一次性
runtime、长期 service、run 溯源和后续更多 job kind。

### 影响范围

- 影响 Eval Bench 的 job 创建、worker 执行、dashboard 使用流程和文档。
- 不影响 `src/shaft` 训练主链、online eval 或既有 checkpoint。
- 旧扁平 job payload 仍通过兼容转换进入 manifest 解析层。

### 修复

- 新增 manifest-driven job spec：`runtime` 管模型后端生命周期，`eval` 管任务、prompt、generation
  和数据参数。
- `eval_job` 支持 `runtime.mode=ephemeral`：worker 启动 job 专属 vLLM OpenAI server，等待 ready，
  执行推理/解析/evaluate，最后关闭进程；runtime log 写入 `runs/<run_id>/logs/runtime.log`。
- `runtime.mode=existing_service` 保持连接已有 endpoint，不负责启停长期服务。
- 前端 Jobs 页改为模板 + JSON manifest 编辑 + `Validate` preflight；后端 preflight 检查
  benchmark/model/task/prompt，并展示将要执行的 vLLM 命令。
- 未知 `runtime.args` 保留为 vLLM CLI flags，避免每新增一个 vLLM 参数都要修改表单 schema。
- 创建 job API 在入队前执行 preflight；当前 UI 只暴露已可执行的 `eval_job`，避免未接入 worker 的
  job kind 进入队列。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m compileall -q projects/eval_bench/eval_bench`
- `PYTHONPATH=projects/eval_bench pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_worker.py projects/eval_bench/tests/test_services.py projects/eval_bench/tests/test_dashboard.py`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 新增 job kind 时，必须同时补：manifest schema、preflight、worker 执行路径、artifact 落盘规则、
  dashboard 模板和测试。
- 长期模型服务继续用 Service registry 管理；单次任务自带的 vLLM 后端继续用 Job runtime 管理。
- Dashboard 表单只生成初始 manifest，不作为参数真源；run manifest 和 job metadata 才是复盘真源。

## 2026-05-10: Eval Bench prompt 模板进入持久化配置层

### 现象

Job manifest 已经可以自由编辑 runtime / eval 参数，但 prompt 仍只能在 JSON 中手写 `prompt_id`、
`prompt_path` 或 inline 文本。前端不能选择 prompt，后端 preflight 也没有把 prompt 模板作为
数据库状态管理，导致任务参数模板不完整，后续新增 prompt 或做 prompt A/B eval 时容易依赖聊天记录
或临时 JSON。

### 根因

Prompt 是 eval spec 的核心输入，但上一版只把 job 和 service 放进 SQLite registry。Prompt 默认值
仍散落在 repo YAML、job template 和 worker fallback 里，没有形成 dashboard、preflight、worker
共享的配置层。

### 影响范围

- 影响 Eval Bench 的 Jobs 页、job preflight、worker 执行和 run 溯源。
- 不影响 `src/shaft` 训练主链。
- 旧 job payload 仍兼容；没有 prompt template 时仍可通过 inline prompt 或 prompt path 执行。

### 修复

- 新增 SQLite `prompt_templates` 表，字段包含 `prompt_id`、label、task、system/user prompt、
  parser、metric profile、visualization profile、generation、data 和 metadata。
- Dashboard 启动时从 `configs/prompts/grounding_layout.yaml`、`grounding_arrow.yaml`、
  `keypoint_arrow.yaml` 种子化默认模板：`grounding_layout.latest`、`grounding_arrow.latest`、
  `keypoint_arrow.latest`。
- `job_spec` preflight 接收 prompt template map，按 `prompt_id` 把 prompt 文本、parser、metric、
  generation 和 data 默认值写入 resolved manifest；显式 manifest 字段优先。
- 创建 job 时保存 resolved payload + resolved manifest，避免 worker 后续依赖前端临时状态。
- Worker 处理队列时也读取同一份 prompt template registry，保证 CLI/API 创建的 job 与 Dashboard
  行为一致。
- Jobs 页新增 prompt template 选择、应用和“从当前 manifest 保存为模板”入口。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_database.py projects/eval_bench/tests/test_worker.py`
- `cd projects/eval_bench/frontend && npm run build`

### 后续防线

- 新增 prompt 不得只改前端 JSON 示例；必须进入 prompt template registry 或 repo prompt YAML 种子。
- Job preflight、worker 和 run manifest 必须看到同一份 resolved prompt 文本和推理参数。
- 允许用户自由添加 manifest 字段，但固定表单/模板只能作为初始值，不得成为第二套参数真源。

## 2026-05-10: Eval Bench 检查器交互与可视化偏好收口

### 现象

Eval Bench 的基准集检查、评测记录检查和工作台设置之间存在体验不一致：基准集页面的图像容器不能
缩放和平移，样本跳转输入框价值很低，样本列表和对象列表大量文本被省略号截断；滚轮缩放过于灵敏，
且无法缩小到 100% 以下；工作台设置里的叠图颜色没有稳定传递到检查器；所有正常实例几乎只按
GT/Pred 着色，无法按 label 调色；箭头 linestrip 只画骨干，没有端点和方向提示。评测记录表的
选择列还依赖浏览器原生 checkbox 外观，在部分环境下会在 checkbox 旁边出现无语义的小点。

### 根因

前端检查器实现出现了双轨：benchmark viewer 和 run viewer 各自维护画布能力，叠图偏好也没有
沉到单一 hook。布局层仍是固定 grid，无法像工程工作台一样调整左侧样本列表和右侧检查器比例。
表格和样本卡片过度使用 `white-space: nowrap` 与 ellipsis，导致真实路径、标签和对象诊断信息
在工程场景下不可读。叠图颜色只覆盖状态层，缺少 label 层颜色真源和方向几何表达。

### 影响范围

- 影响 Eval Bench dashboard 的基准集检查、run 样本检查、工作台设置和评测记录表。
- 不影响后端 evaluator、metric 计算和 run artifact 格式。
- 不影响 `src/shaft` 训练主链。

### 修复

- 抽出共享 `CanvasStage`：benchmark 和 run 检查器统一使用同一套等比例自适应、滚轮缩放、拖拽
  平移和图片预加载逻辑。
- 缩放范围改为 25% 到 800%，滚轮灵敏度降到原先的低敏级别，缩小后仍允许拖拽查看画布位置。
- 新增可复用 `ResizableSplit`，用于样本列表/主画布、主画布/对象检查器的可拖拽分栏，并持久化
  用户调整后的宽度。
- `ResizableSplit` 继续推广到 Compare 工作区、成对样本对比、工作台设置和 job manifest 编辑/预检查区，
  避免这些工程面板继续使用固定比例 grid。
- 分栏可拖拽范围扩大，并由容器宽度动态限制上限，避免稍微拖动就被固定 `maxSize` 锁死。
- 新增 `frontend/src/workspaceSettings.ts` 作为浏览器侧用户设置真源，集中管理 label × role 颜色、
  叠图样式、鼠标/滚轮交互参数、快捷键、缩放上下限、sidebar 折叠状态和分栏尺寸读取。
- 工作台设置页的叠图预览复用正常 `CanvasStage`；预览底图可以是稳定 sample asset，因此预览也
  支持滚轮缩放、拖拽平移、GT/Pred 叠图、方向三角形和运行时 label 颜色匹配。
- 左侧导航栏新增图标按钮控制收起/展开，折叠态保留 icon-only 导航并持久化。
- 删除样本跳转输入框，保留列表选择、分页和可配置的样本前后切换 action。
- 叠图偏好统一由 `useWorkspaceSettings` 管理，工作台设置和检查器共享线宽、点大小、
  标签字号与按 label × role 的颜色配置。
- label 颜色不再内置 `arrow/icon/text` 等任务名；用户可手动添加任意 label × role 颜色规则，运行时按
  实际 label 和 role 匹配，未配置时走固定 role 默认色。
- arrows/linestrip 增加起点、终点和位于中间线段的自适应方向三角形；误检/漏检使用 `FN` / `FP`
  role，正常实例使用 `GT` / `Pred` role，再叠加显式 label × role 颜色。
- 画布拖动灵敏度恢复为 1:1，滚轮缩放灵敏度调到当前低敏基线的 4 倍，兼顾局部排障速度和可控性。
- 评测记录选择框改为显式 checkbox 样式，去掉浏览器原生外观造成的无语义小点。
- 样本卡片和对象列表改为可换行排版，避免关键路径、label、IoU/诊断文本被省略号隐藏。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests/test_job_spec.py projects/eval_bench/tests/test_dashboard.py`
- Dashboard 页面渲染冒烟：基准集检查、run 检查、工作台设置、Compare 和 Jobs 页面。
- 专项交互冒烟：sidebar 收起/展开、Settings 分栏大范围拖拽、动态 label 颜色添加与持久化。

### 后续防线

- 新增可视化页面不得再复制一套独立画布；默认复用 `CanvasStage` 和 `useWorkspaceSettings`。
- 用户偏好不得继续散落在页面组件里；新增设置必须优先进入 `workspaceSettings.ts`，再由页面消费。
- 任何工程检查列表默认不能用无条件 ellipsis 隐藏关键信息；必要时用可换行、title 或详情面板。
- 新增颜色/样式偏好必须同时在工作台设置、benchmark viewer 和 run viewer 中生效。
- 表格选择、状态点等 UI 元素必须有明确语义；纯浏览器默认外观导致的装饰性小点要用显式样式收口。

## 2026-05-11: Eval Bench prompt 目标 label 语义与 Compare 页面收口

### 现象

使用 `grounding_layout.latest` prompt 在多任务 benchmark 上跑 eval 后，报告里几乎没有 arrow 预测结果，
容易被误判为模型完全不会检测 arrow。进一步查看 run manifest 和 prompt 文本发现，这个 run 实际使用的是
layout prompt，只要求输出 `icon / image / shape`；但 benchmark GT 同时包含 arrow，因此旧 evaluator
把未要求输出的 arrow 当成了漏检。Compare 页面也只展示整体 delta 和样本表格，无法清楚说明当前 run
实际评价了哪些 label；top 改善/退化样本表在窄列下大量显示省略号，排障价值不足。

### 根因

Eval Bench 之前把 `task=detection` 误当成完整评测语义，而没有把 prompt 的目标 label 集合纳入
resolved manifest 和 `EvalSpec`。多任务 benchmark 下，`task=detection` 只能说明对象类型是 bbox，
不能说明本次 prompt 应该覆盖 layout、arrow 还是两者。前端 compare 页面同样没有展示 target label
约束，导致用户无法从报告上区分“模型没预测 arrow”和“本轮 layout prompt 没要求预测 arrow”。

### 影响范围

- 影响 Eval Bench 的 eval job、worker、evaluator、comparison report 和 dashboard compare 页面。
- 影响多任务 benchmark 上的 layout/arrow 分任务评测解释；可能造成 arrow recall 被错误归因。
- 不影响 raw prediction snapshot 的保存格式，也不影响 Shaft 训练主链。

### 修复方式

- Prompt template registry 为 `grounding_layout.latest`、`grounding_arrow.latest` 和
  `keypoint_arrow.latest` 增加 `target_labels` 元数据。
- Job manifest 默认值、prompt template 应用、preflight payload 和 worker 创建 `EvalSpec` 时都传递同一份
  `target_labels`，避免前端、后端和 worker 各自推导。
- Evaluator 在匹配前按 `target_labels` 同时过滤 GT 和 prediction，并把目标 label 写入 report summary。
- Comparison report 写入 target label、per-label delta 和语义 warning；Dashboard Compare 页展示目标
  label chip、warning、分 label delta，以及更可读的 top 改善/退化样本条目。
- 工作台设置的叠图预览继续复用真实 viewer，新增标签底色透明度、bbox 填充透明度和方向箭头大小等可调项，
  让设置页能实际观察 GT/Pred、FN/FP、label 色和 arrow 方向效果。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- Dashboard render check：
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/settings INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/compare npm run render-check`

### 后续防线

- 新增 prompt template 必须声明 `target_labels`；多任务 benchmark 上不能只靠 `task` 推断评测 label。
- 评测报告和对比报告必须显示实际 target label，避免把 prompt 语义问题误判成模型能力问题。
- Compare 页面不能只给整体分数；至少要保留 per-label delta、样本级改善/退化入口和目标 label 提示。
- 工程排障表格不能把关键 sample id、图片名、TP/FP/FN/IoU 全部压成省略号；窄列下优先使用专用条目或换行布局。

## 2026-05-11: Eval Bench detail 慢加载、真实预览和 Job/Run 工作区统一

### 现象

Benchmark inspector / run inspector 切换样本时经常出现“正在加载样本详情”的空白状态；切换到
inspector 页面后左侧导航会被 CSS 自动压缩，表现和用户手动折叠不同；工作台设置页的预览仍是硬编码
假样本；Jobs 和 Runs 分散在不同页面，用户需要在任务记录和评测结果之间来回跳转。Compare 页面虽然有
整体 delta，但从报告跳到具体改善/退化样本仍不够直接。

### 根因

前端先请求 sample page，再按当前 index 单独请求 detail，切样本时没有保留上一份 detail，也没有预取邻近
样本详情。后端 `store` 每次 detail/image 请求都会重新读取并展开 split manifest。样式层用
`.app-shell:has(.visual-inspector-page)` 强制修改全局 grid，绕过了 `useSidebarPreference` 的唯一状态源。
设置页为了演示 overlay 复制了一份固定图和固定实例，没有复用 benchmark 真值样本。Jobs 和 Runs 在信息架构
上分裂，缺少 job 成功后到 run snapshot 的直接关系。

### 影响范围

- 影响 Eval Bench dashboard 的 benchmark inspector、run inspector、settings、jobs 和 compare 页面。
- 不影响 prediction snapshot、metric 计算结果和 Shaft 训练主链。
- 旧截图或浏览器缓存可能继续加载旧 Vite asset，dashboard 重启后会读取新的构建产物。

### 修复方式

- `EvalBenchStore` 增加 benchmark/run split JSON 路径缓存，避免重复读取 split manifest。
- 前端 sample detail query 使用上一份 detail 作为 placeholder，并预取当前样本前后邻近详情；label 过滤后
  以真实 `sample.index` 作为选择真源，不再把过滤列表位置误当成全局 index。
- 删除 inspector 页面触发的全局 `:has(.visual-inspector-page)` 自动收缩 CSS，只保留用户手动折叠导航。
- 新增 `/api/settings/preview-sample`，从现有 benchmark 选择带 bbox/linestrip/keypoints 的真实样本给设置页
  预览；设置页仍复用 `CanvasStage` 和统一 overlay 配置。
- `评测中心` 改为统一活动流：左侧 job 队列，右侧最近结果卡片；job 详情在已有 `run_id` 时提供“打开结果”
  链接。完整 run 管理仍在 `结果库`。
- Compare 报告增加 target label 过滤入口、首个改善/退化样本快捷入口，并在 top sample 中展示样本涉及的
  label delta。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:metrics`
- Dashboard render check：
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/settings INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/benchmarks/multitask_val_v1 INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/jobs INTERACTION_SMOKE=1 npm run render-check`
  - `EVAL_BENCH_URL=http://127.0.0.1:8767/compare INTERACTION_SMOKE=1 npm run render-check`

### 后续防线

- 任何需要跨页面共享的 UI 状态必须走显式 state/hook，不允许用页面级 CSS selector 改全局布局状态。
- 检查器页面切换样本不能默认清空主画布；至少保留旧 detail、预取邻近样本或展示可定位的轻量刷新提示。
- Settings preview 必须尽量使用真实 benchmark/run 样本；只有 store 里没有可绘制样本时才允许 fallback 示意图。
- Jobs 是执行记录，Runs 是结果快照；前端可以统一入口，但数据语义不能合并成同一个对象。

## 2026-05-12: Eval Bench 设置页玩具化界面重构

### 现象

工作台设置页虽然能调颜色、线宽和鼠标参数，但视觉上仍像临时表单：卡片感重、控件堆叠、缺少稳定
配置键名，和 VS Code 这类工程软件的 Preferences 体验差距明显。

### 根因

早期实现把设置页当成演示面板处理，直接复用 `WorkspaceTabs`、卡片和通用控制组，没有把“配置项真源”
作为页面的一等对象展示。第一轮改成三栏 Preferences 后虽然更像工程设置页，但右侧预览仍被配置列挤压，
不符合 Eval Bench 以可视化检查为主的产品目标。

### 影响范围

- 影响 Eval Bench dashboard 的 `工作台设置` 页面可用性和长期维护性。
- 不影响评测 job、metric 计算、prediction snapshot 或训练主链。

### 修复方式

- 设置页改为顶部 command bar + 大画布预览 + 底部 Preferences 抽屉：学习 VS Code 的设置键名和搜索，
  但把主空间让给 FiftyOne/CVAT 式的视觉检查区域。
- 顶栏和设置页增加轻量 local profile 展示，明确当前阶段是浏览器本地用户偏好，而不是权限系统。
- 每个设置行展示稳定配置键名，例如 `evalBench.overlay.colors`、`evalBench.viewer.interaction`。
- 颜色控件改为紧凑 token 行并支持自适应换行，避免窄宽度下挤压和溢出。
- 保留真实 benchmark sample 预览，并继续复用统一 `CanvasStage`、`workspaceSettings` 和 overlay 配置层。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8774/settings npm run test:settings-preview`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8774/runs/config_smoke_prompt_params?perf=1 npm run test:viewer-performance`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8774/settings SCREENSHOT_PATH=/home/tanjingyuan/code/arrow-vlm/temp/eval_bench_settings_console_layout.png INTERACTION_SMOKE=1 npm run render-check`

### 后续防线

- 新增用户设置时必须先确认配置键名和所属分组，再落 UI 控件。
- 设置页不再使用大块卡片堆表单，也不再使用三栏压缩预览；默认采用命令栏、设置键名、底部抽屉和大画布预览。
- 控件必须在 1440px 级别工作区内完整显示，不能依赖页面整体滚动才能看到关键配置。

## 2026-05-25: Eval Bench rank board 与控制台 UI 语义收敛

### 现象

Eval Bench 的 Rank Board 默认使用加权综合分，并在前端显示综合分公式；多个页面把高级检索控件直接铺开，
导致主工作区被 select 和说明文字占用。总览和 Compare 页面还出现面向开发过程的解释性文字。Run
样本列表在每张图片条目下显示 TP/FP/FN，和用户实际需要的快速浏览信息不一致。可视化检查器切换样本时，
label 和图层偏好会被重置。

### 根因

Rank Board 早期为了快速排序把 precision、recall 和 mIoU 组合成 `score`，没有把主指标选择建模为一等排序语义。
`AdvancedFilterBar` 同时承担摘要和完整表单，缺少折叠状态。样本检查器把样本切换当成全新 viewer 初始化，
因此把可复用的审阅偏好和当前样本的临时 hover/lock 状态混在了一起。

### 影响范围

- 影响 Eval Bench dashboard 的 Rank Board、Overview、Compare、Runs/Benchmarks/Jobs/Services 的筛选入口和 run sample viewer。
- 影响 `/api/rank-board` 与 CLI `rank-board` 的默认排序语义。
- 不影响 metric report 本身、prediction snapshot 或 Shaft 训练主链。

### 修复方式

- Rank Board 默认主指标改为 `f1_iou50`；`score` 仅作为兼容字段镜像 F1，不再表示加权综合分。
- CLI/API/前端排序入口使用 `f1_iou50` 作为默认值，并保留 precision、recall、mIoU、预测数、创建时间和 run id 排序。
- Rank Board、总览和 Compare 的排行榜入口改用 Eval Bench 自有 `metrics` 图标，避免 lucide trophy 混入系统图标风格。
- `AdvancedFilterBar` 改为默认折叠的 Filter 入口，展开后显示检索表单；UI 中不再显示长段解释性 meta 文案。
- 总览移除 block 内说明文字，最近 run 列表按条目数量计算列数，并压缩为可扫描的编号条目。
- 顶栏 local profile 与在线状态改成同一组呼吸感状态 capsule。
- Run 样本列表只显示 `真实 N / 预测 N`；切换样本时只清理 hover/lock，label 和图层偏好写入浏览器本地状态，
  翻页后继续沿用当前审阅视图。
- 删除 run/job/service 的原生 `confirm()`，统一为 `DangerConfirmDialog`。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_exposes_independent_rank_board projects/eval_bench/tests/test_cli.py::test_cli_prints_filtered_rank_board`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:dialogs`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`
- `cd projects/eval_bench/frontend && npm run test:workspace-settings`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ EVAL_BENCH_RUN_ID=eval_20260513_103418_ebb7f052 npm run test:shortcuts`
- Playwright smoke：在真实 run inspector 上关闭预测图层、切换 label、翻到下一页，确认左侧条目只显示
  `真实 N / 预测 N`，且预测图层偏好没有恢复默认。
- `curl -fsS 'http://127.0.0.1:8766/api/rank-board?limit=1'` 确认返回 `primary_metric=f1_iou50` 和 `f1_iou50` entry 字段。

### 后续防线

- Rank Board 不能重新引入默认加权综合分；如果需要 bench/metric 加权方案，必须作为显式方案配置和独立展示，不得覆盖默认 F1 主指标。
- 页面级 filter 入口默认折叠；新增页面不能把多组 select 平铺到主工作区。
- 可视化检查器的持久偏好和样本级临时交互状态必须分离，翻页不能重置用户已选择的显示偏好。

## 2026-05-25: Eval Bench detection label 子任务创建入口收口

### 现象

Detection 的 label 子任务已经能在后端通过 `target_labels`、prompt metadata 和 `label_policy.py` 表达，
但新建评测任务面板主要依赖用户直接编辑 JSON manifest。前端拿不到 benchmark summary 的 label 索引，
导致人类创建 label 子任务时没有稳定候选，agent CLI 也无法通过 `list-benchmarks` 直接看到 benchmark
可用 label。

### 根因

Benchmark manifest 写入了 `labels`，但 `BenchmarkSummary` 没有把这组 label 暴露为列表真源。任务创建 UI
只能从 prompt template 或用户手写 manifest 推断 target labels，形成“后端语义可用、创建入口不够显式”的断层。

### 影响范围

- 影响 Eval Bench Jobs 页新建 detection 子任务的可用性。
- 影响 agent 通过 `list-benchmarks` 枚举 benchmark 可用 label 的稳定性。
- 不影响已创建 run 的评估语义，也不影响 metric report、sample scope 或 Shaft 训练主链。

### 修复方式

- `BenchmarkSummary` 增加 `labels` 字段；manifest 缺少 labels 时，store 通过 sample scan fallback 补齐。
- 前端 `BenchmarkSummary` 类型同步增加 `labels`。
- Jobs 页新建评测面板增加 Detection 子任务 chips，从 benchmark labels、prompt template target labels 和当前
  manifest target labels 合并候选；点击 chip 会直接更新 manifest 的 `eval.target_labels`。
- `全部候选` 会显式写入当前候选集合；`默认策略` 会删除 manifest 中的 `target_labels` 字段，让后端继续按统一
  label policy 解析默认范围。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_api_exposes_store_state projects/eval_bench/tests/test_dashboard.py::test_dashboard_creates_benchmark_copy_from_raw_data`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py::test_cli_lists_benchmarks_runs_and_comparisons_with_agent_filters`
- `curl -fsS http://127.0.0.1:8766/api/state` 和 `/api/benchmarks?limit=1` 确认 benchmark summary 返回
  `labels=["arrow","icon","image","shape"]`。
- Playwright smoke：打开 Jobs 新建评测弹窗，点击 `icon` chip 后 manifest 出现 `target_labels=["arrow","icon"]`，
  点击 `全部候选` 后写入全部候选 label，点击 `默认策略` 后 manifest 删除 `target_labels`。
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- 新增子任务入口时，候选项必须来自 store/API/CLI 的正式字段，不能让前端扫描 artifact 文件。
- 留空 target labels 的语义继续归 `label_policy.py`，前端只负责显式写入用户选择的 `target_labels`。

## 2026-05-25: Eval Bench agent 模板发现与 prompt registry CLI 闭环

### 现象

Dashboard 可以通过 `/api/job-templates` 和 `/api/prompt-templates` 发现 job/prompt 模板，也能在 UI 中保存
prompt template；但 CLI 没有对应入口。Agent 如果要创建 manifest-first job，仍需要知道前端 API 或直接读
SQLite / 前端状态，和“agent 不做 hack”的目标不一致。

### 根因

早期 CLI 优先补了 job 入队、run/rank/benchmark/comparison 查询和 run note，但 prompt template registry
的读写仍只暴露在 dashboard API。模板发现和 prompt registry 管理没有进入 agent-safe CLI surface。

### 影响范围

- 影响 agent 自动创建 eval job 前的模板发现、prompt 选择和 prompt template 维护。
- 不影响已有 dashboard UI、job worker、metric report 或训练主链。

### 修复方式

- 新增 `list-job-templates`，直接输出 `job_spec.job_templates()`。
- 新增 `list-prompt-templates --task --query --offset --limit`，复用 `EvalBenchDatabase.list_prompt_templates()`。
- 新增 `upsert-prompt-template --payload-json/--payload-file` 和 `delete-prompt-template --prompt-id`，
  CLI 与 dashboard API 共用同一个 prompt template registry。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py::test_cli_manages_job_and_prompt_templates_for_agents`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_api_exposes_store_state`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run python -m compileall -q projects/eval_bench/eval_bench scripts/eval_bench.py`
- `uv run python scripts/eval_bench.py list-job-templates --query keypoint`
- `uv run python scripts/eval_bench.py list-prompt-templates --task detection --query arrow --limit 2`

### 后续防线

- 新增 dashboard 可操作对象时，必须同步检查是否需要 agent-safe CLI 发现和维护入口。
- Prompt template 的唯一真源是 database registry；前端和 CLI 只能通过同一 registry 读写。

## 2026-05-25: Eval Bench Overview 实时总控信号收口

### 现象

总览页已经移除了长说明文字和精细指标，但页面仍更像静态汇总面板。用户需要它承担实时控制台职责：
只显示系统运行态、数据规模和近期写入节奏，同时保持单屏可读，不把排行榜或对比页的细指标搬回来。

### 根因

Overview 只消费 `/api/state` 的 run/benchmark 粗汇总，缺少 job queue、service 和 scheduler 这类实时运行态。
写入节奏也只按已有 run 日期聚合，缺少连续日期桶，导致稀疏数据时空间利用不稳定。

### 影响范围

- 影响 Eval Bench dashboard 的 Overview 信息架构和实时可观测性。
- 不影响 Rank Board 的默认 F1 排序、run sample viewer、job worker 或评测报告语义。

### 修复方式

- Overview 增加 telemetry strip，复用现有 jobs、services 和 scheduler API，展示 scheduler、queued jobs、
  running jobs、service live count 和 job records。
- Run 写入节奏改为以最新 run 日期为右边界的连续 12 个日期桶，没有写入的日期显式显示为 0。
- Overview 保持粗粒度控制台视角，不展示 precision、recall、mIoU 等精细评测指标。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- Playwright smoke：打开 `http://127.0.0.1:8766/`，确认 `.overview-telemetry-panel` 存在、
  telemetry cell 数为 5、timeline 日期桶数为 12，且总览页正文不包含 `precision` / `recall`。

### 后续防线

- Overview 新增状态只能接入已有 store/API/CLI 真源，不能在页面层复制 scheduler、job 或 service 状态机。
- Overview 继续只承载总控信号；精细指标和排行策略留在 Rank Board 与 Compare。

## 2026-05-25: Eval Bench 顶栏状态胶囊与排行榜 icon 语义收口

### 现象

顶栏 `Profile local / 同步中` 状态区仍是硬边框小块，和当前工作台的轻量状态设计不一致；Rank Board
的“入榜”“已评估”和页面入口“排行榜”复用同一个 metrics icon，导致三个不同语义在视觉上无法区分。

### 根因

早期 dashboard icon 只按“指标/结果”粗分类复用资产，没有为 rank board 页面入口、rank entry 计数和 evaluated
run 计数拆出语义 key。顶栏状态也只复用了通用 pill 样式，没有区分在线、同步中和异常的视觉节奏。

### 影响范围

- 影响 Eval Bench dashboard 顶栏状态识别和 Rank Board 指标卡识别。
- 不影响 API、CLI、metric report 或评测语义。

### 修复方式

- `iconLibrary.tsx` 增加 `rankBoard`、`rankEntry` 和 `evaluatedRun` 语义图标。
- 侧栏、总览快捷入口、Compare 的排行榜入口统一使用 `rankBoard`。
- Rank Board 的“入榜”使用 `rankEntry`，“已评估”使用 `evaluatedRun`。
- 顶栏 profile/status 区改为圆角 capsule 组；同步中状态增加克制的 breathing 动效和状态点。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- 新增 dashboard 图标时必须先补 `AppIcon` 语义 key；不同业务状态不能只因为都属于指标域而复用同一个图标。
- 顶栏运行态优先使用状态 capsule，不在页面中散落独立同步文字。

## 2026-05-25: Eval Bench Rank Board 显式加权排行入口

### 现象

Rank Board 默认排行已经改为 F1 主指标，但仍缺少一个 agent-safe 的显式加权方案入口。用户需要在不改变默认
F1 排行语义的前提下，按指定 benchmark 和 metric 权重得到一个最终分，并能解释每条 entry 的分数来源。

### 根因

早期实现把 `score` 固定为 F1 兼容字段，只允许在已有单项 metric 之间切换排序。加权排行还停留在设计约束，
没有进入 store/API/CLI 的正式参数，也没有返回可审计的贡献项。

### 影响范围

- 影响 Eval Bench Rank Board 的 API/CLI 排序能力和 agent 自动排行能力。
- 不影响默认 `/api/rank-board`、CLI `rank-board` 的 F1 排行结果。
- 不影响 evaluator report、prediction snapshot 或 run note。

### 修复方式

- `EvalBenchStore.rank_board()` 增加显式 `rank_scheme` 输入；没有 scheme 时仍返回 `primary_metric=f1_iou50`。
- `rank_scheme` terms 要求包含 `benchmark_id`、`metric`、`weight` 和 `missing`，缺失策略支持
  `drop`、`skip`、`zero`。
- 显式 weighted scheme 会返回 `primary_metric=weighted_score`、原始 `rank_scheme`、`score_formula` 和
  entry-level `score_components`。
- `/api/rank-board` 接受 JSON string 形式的 `rank_scheme` query 参数，非法 scheme 返回 400。
- CLI `rank-board` 增加 `--rank-scheme-json` 和 `--rank-scheme-file`，供 agent 直接传入方案。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m compileall -q projects/eval_bench/eval_bench projects/eval_bench/tests/test_dashboard.py projects/eval_bench/tests/test_cli.py`
- `.venv/bin/ruff check projects/eval_bench/eval_bench/store.py projects/eval_bench/eval_bench/cli.py projects/eval_bench/eval_bench/dashboard.py projects/eval_bench/tests/test_cli.py projects/eval_bench/tests/test_dashboard.py`
- Store 直调 smoke：构造两个 succeeded run，确认默认 board 仍是 `f1_iou50`，显式 scheme 返回
  `weighted_score`、原始 `rank_scheme` 和 `score_components`。
- `cd projects/eval_bench/frontend && npm run build`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=projects/eval_bench timeout 90 .venv/bin/pytest -q projects/eval_bench/tests/test_dashboard.py::test_dashboard_exposes_independent_rank_board projects/eval_bench/tests/test_cli.py::test_cli_prints_filtered_rank_board`
  当前机器上该 focused pytest 在 CLI/dashboard 重依赖 import 阶段超时无输出，未拿到 pytest 结果；后续需要在
  import 性能恢复后重跑。

### 后续防线

- 加权排行只能通过显式 scheme 启用；不能把 weighted score 回写成默认 `score` 语义或覆盖默认 F1。
- 新增 weighted metric 时必须同时补 `score_components`，保证 agent 能解释最终分数。

## 2026-05-25: Eval Bench CLI import 性能与总览密度收口

### 现象

Rank Board 加权排行完成后，focused pytest 一度卡在 CLI/dashboard import 阶段；同时总览页的
`Run 写入节奏` 作为大块 card 占用了过多区域，最近 run 条目会被剩余高度拉伸，顶栏 `在线`
状态也被通用 pill 规则覆盖成硬角。

### 根因

`cli.py` 顶层直接导入 dashboard、worker、evaluator、store、service 等模块，dashboard/orchestrator
顶层又导入 worker，进而拉起 Shaft、Transformers、NumPy 等重依赖。前端层面，写入节奏和最近 run
都按大 panel 处理，没有根据数据量自适应空间；`design.css` 后段的通用 badge 规则重新覆盖了
`.status-pill` 的圆角。

### 影响范围

- 影响 Eval Bench agent-safe CLI 查询、dashboard import、focused pytest 稳定性。
- 影响 Dashboard 总览页信息密度、写入节奏可读性和顶栏在线状态视觉。
- 不影响 evaluator 指标、rank board 默认 F1 语义、weighted scheme 或 worker 执行结果。

### 修复方式

- CLI 顶层只保留 `DEFAULT_STORE_ROOT` 轻量依赖；各命令在执行时局部导入对应 store、database、
  dashboard、worker、evaluator 或 service 模块。
- Dashboard 和 orchestrator 移除顶层 worker import；process-next、cancel runtime 和后台 job 执行处按需导入。
- 总览页把 `Run 写入节奏` 改成 12 桶微型柱状条，新增生命周期、任务类型、模型分布和 Prompt 分布
  四张环形占比 + 条形图矩阵；最近 run 改成内容自适应紧凑行。
- 顶栏 profile/status 去掉外层圆角 wrapper，`在线` / `同步中` 的圆角和 breathing 动效固定在
  `.status-pill` 自身，并避免被通用 badge 规则覆盖。

### 回归测试

- `PYTHONPATH=projects/eval_bench .venv/bin/python -m compileall -q projects/eval_bench/eval_bench/cli.py projects/eval_bench/eval_bench/dashboard.py projects/eval_bench/eval_bench/orchestrator.py`
- `.venv/bin/ruff check projects/eval_bench/eval_bench/cli.py projects/eval_bench/eval_bench/dashboard.py projects/eval_bench/eval_bench/orchestrator.py`
- `PYTHONPATH=projects/eval_bench .venv/bin/pytest -q projects/eval_bench/tests/test_cli.py projects/eval_bench/tests/test_dashboard.py::test_dashboard_exposes_independent_rank_board projects/eval_bench/tests/test_dashboard.py::test_dashboard_api_exposes_store_state`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`
- Playwright CSS 断言：`.topbar-actions` 无 border/background/shadow，`.status-pill` radius 为 `999px`，
  总览有 12 个 rhythm bar、4 个 mini chart，最近 run 行高为 54px，页面无全局滚动。
- Import timing smoke：`cli_import_seconds=0.028`，`dashboard_import_seconds=0.303`。

### 后续防线

- Agent-safe CLI/API 入口不得在模块顶层导入 dashboard、worker、evaluator 或模型运行时重依赖。
- 总览只承载高密度运行态可视化；低频条目必须内容自适应，不能按剩余高度拉伸。
- 顶栏状态样式只由 status pill 承载，通用 badge/chip 规则不能覆盖其圆角和动效。

## 2026-05-25: Eval Bench Overview 视觉约束固化到 layout smoke

### 现象

总览页和顶栏状态完成视觉收口后，关键约束仍主要依赖一次性 Playwright 断言和人工截图复查。
后续修改 `styles.css` 或 `design.css` 时，仍可能把 `在线` 状态重新覆盖成硬角，或让最近 run / 写入节奏重新拉伸成低密度大块。

### 根因

`test:layout` 已覆盖全局滚动、弹窗边界、高级检索和独立 chunk，但没有把 Overview 的高密度信息架构
和顶栏 status capsule 作为正式验收项。测试只等待 `.overview-rhythm-strip` / `.overview-mini-chart`
存在，不能证明它们没有退回旧的大块 timeline 语义。

### 影响范围

- 影响 Eval Bench dashboard 的 Overview 密度回归防线和顶栏状态视觉稳定性。
- 不影响后端 API、CLI、rank board 排序或 evaluator 语义。

### 修复方式

- `layout-smoke-check.mjs` 新增 `assertTopbarStatus()`，所有页面都检查 `.topbar-actions` 没有外层容器样式，
  `.topbar .status-pill` 保持圆角 capsule 和隐藏溢出。
- `layout-smoke-check.mjs` 新增 `assertOverviewDensity()`，Overview 额外检查 12 个 rhythm bar、至少 4 个
  mini chart、最近 run 行不超过 72px、旧 timeline markup 不再出现、写入节奏条保持紧凑。
- README 和 `docs/scripts.md` 补充 `test:layout` 的 Overview / 顶栏验收范围。

### 回归测试

- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- Overview 或顶栏样式变更必须继续跑 `test:layout`；如果有新的高密度控制台模块，也应把关键密度约束放入该 smoke。

## 2026-05-25: Eval Bench inspector 样本筛选收敛到高级检索

### 现象

Runs、Benchmarks、Jobs、Services、Compare 和 Rank Board 已经复用 `AdvancedFilterBar`，但 Benchmark
Inspector 和 Run Inspector 的样本侧栏仍直接堆叠 `FilterSelect`。这和“论文检索式高级检索”的交互目标不一致，
也会在窄侧栏里继续增加固定高度。

### 根因

样本检查器最早只需要 label/error 两个条件，所以保留了局部 `sample-filters` 容器；后续全站筛选体系升级后，
该局部实现没有回收到统一组件，layout smoke 也没有进入有数据的 inspector detail route 检查筛选形态。

### 影响范围

- 影响 Benchmark Inspector / Run Inspector 样本侧栏的信息密度、筛选一致性和窄屏滚动边界。
- 不影响后端 sample API、label scope、metric report 或 viewer 几何渲染。

### 修复方式

- Benchmark Inspector 的 label 筛选改为折叠式 `AdvancedFilterBar`。
- Run Inspector 的 error + label 筛选改为折叠式 `AdvancedFilterBar`。
- 删除旧 `.sample-filters` 样式，并为 inspector 侧栏增加紧凑版 advanced filter 样式。
- `layout-smoke-check.mjs` 动态读取 `/api/state`，在有 benchmark/run 数据时加入 inspector detail route，
  并检查 inspector 样本筛选必须使用 `AdvancedFilterBar`、默认折叠、不能溢出侧栏。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 新增样本级筛选条件也要进入 `AdvancedFilterBar`，不能恢复局部 `sample-filters` 容器。
- 有真实 store 数据时，layout smoke 必须继续覆盖 benchmark/run inspector detail route。

## 2026-05-25: Eval Bench Overview 高密度图表矩阵

### 现象

总览页已经压缩了写入节奏和最近 run，但主内容区仍只有 4 个 mini chart，最近 run 面板占用一整列。
在 run 和 benchmark 数量不多时，页面仍没有把一个屏幕内的空间转换成足够多的实时运营信号。

### 根因

Overview 的信息架构仍沿用“图表卡 + 列表卡”的比例，缺少专门面向总控工作台的可视化矩阵。
布局 smoke 也只要求至少 4 个 mini chart，无法阻止后续修改继续把主区域留给低密度列表。

### 影响范围

- 影响 Eval Bench Dashboard 总览页的信息密度和实时运营可读性。
- 不影响 evaluator 指标计算、Rank Board 默认 F1 排序或 weighted rank scheme。
- 总览仍不展示 precision / recall / IoU 细粒度指标，避免和排行榜、对比页职责重叠。

### 修复方式

- Overview 主区域改为 12 个紧凑 mini chart：Run 生命周期、评测覆盖、Run 任务、模型分布、Prompt 分布、
  Benchmark 任务、Label footprint、样本规模、数据层、Split 分布、Label scope 和 Run 新鲜度。
- 最近 run 改成右侧窄事件轨，最多显示 6 条紧凑行，不再占用主图表矩阵。
- 压缩 mini chart 的标题、环形图、条形图和 run 行高度，并限制矩阵行高上限，保证 desktop / compact 视口能在一个控制台区域内承载更多图表。
- `layout-smoke-check.mjs` 把 Overview mini chart 下限提高到 12，并检查总览正文不能出现 precision / recall / IoU 文案。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- Overview 新增内容优先进入粗粒度 mini chart 或 telemetry cell；不要把总览重新变成长列表。
- 细粒度模型能力指标继续放在 Rank Board、Run Inspector 和 Compare，不进入总览。

## 2026-05-25: Eval Bench manifest toolbar select 收敛到基础控件

### 现象

大部分页面筛选已经收敛到 `AdvancedFilterBar`，标准动作也收敛到 `ActionButton` / `CommandButton`，
但评测任务 manifest toolbar 仍在业务页直接写两个 `filter-select compact`：模板选择和 Prompt 选择。
这会让局部输入控件和页面级 filter 控件继续混用同一套样式语义。

### 根因

早期 toolbar 为了快速搭建任务创建表单，直接复用了 filter select 的 CSS class。后续抽出
`controlPrimitives.tsx` 后，number/color/toggle 已经有基础控件，但 select 没有对应 primitive，
导致业务页仍要自己拼 label + select。

### 影响范围

- 影响 Eval Bench Dashboard 任务创建 toolbar 的组件化一致性和后续可维护性。
- 不影响 job manifest 语义、prompt template 应用、target label policy 或后端 preflight。

### 修复方式

- 新增 `CompactSelectControl`，把局部下拉输入纳入 `controlPrimitives.tsx`。
- Manifest toolbar 的模板选择和 Prompt 选择改用 `CompactSelectControl`。
- 修正 manifest toolbar grid 为“两组 select + 四个动作”的稳定列结构，在中窄屏降为两列/单列。
- `test:ui-contracts` 增加约束：Jobs 页不能再直接创建 `filter-select compact`，manifest toolbar 至少使用两个
  `CompactSelectControl`。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`

### 后续防线

- 页面级检索继续走 `filterControls.tsx`；局部 toolbar/input 控件走 `controlPrimitives.tsx`。
- 新增 select 输入时先判断是否是页面级 filter，避免业务页继续复制筛选样式。

## 2026-05-25: Eval Bench Settings 标准动作收敛

### 现象

Manifest toolbar 的 select 已经收敛到 `CompactSelectControl` 后，Settings 页仍有若干标准动作直接写
原生 `button`：搜索清空、重置样式、单个 label 清除、清空 label 颜色和重置交互；预测线型 select
也仍在页面内直接拼 `compact-select dense`。

### 根因

Settings 页面早期作为独立工作台快速迭代，部分按钮具有局部样式但本质仍是标准动作。`test:ui-contracts`
只覆盖了快捷键设置面板，没有覆盖 Settings 主页面里的 reset / clear / select 回流。

### 影响范围

- 影响 Eval Bench Settings 页组件边界一致性和后续维护。
- 不影响浏览器本地设置 schema、viewer 渲染、快捷键 action registry 或 run/benchmark 数据。

### 修复方式

- 预测线型下拉改用 `CompactSelectControl`。
- 搜索清空改用 `IconActionButton`。
- 重置样式、清空 label 颜色、重置交互和单个 label 清除改用 `ActionButton`。
- `test:ui-contracts` 增加 Settings 主页面约束，阻止 `compact-select dense`、原生
  `settings-inline-action` 和 label 清除原生 button 回流。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/settings npm run render-check`

### 后续防线

- Settings 页新增标准动作时必须先走 `ActionButton` 或 `IconActionButton`。
- Settings 页新增局部 select 时必须先走 `CompactSelectControl`，不要继续手写 label + select 外壳。

## 2026-05-25: Eval Bench Settings 页面从 main.tsx 拆出

### 现象

`main.tsx` 已经超过 2800 行，并且仍直接承载 Settings 工作台的预览 query、分组状态、本地偏好 mutations
和整页 JSX。虽然 Settings 控件已经组件化，但页面实现仍堆在主路由文件里，和“main 只做路由和页面装配”的
边界不一致。

### 根因

Settings 页在早期为了快速迭代直接落在 `main.tsx`，后续虽然陆续抽出了 `settingsControls.tsx`、
`workspaceSettings.ts` 和 `controlPrimitives.tsx`，但没有把页面容器本身拆出。

### 影响范围

- 影响 Eval Bench dashboard 前端模块边界、main entry 可维护性和后续页面级重构。
- 不影响 Settings 本地配置 schema、viewer 预览渲染、快捷键 action registry 或后端 API。

### 修复方式

- 新增 `settingsPage.tsx`，承载 Settings 页 query、预览、分组装配和本地偏好 mutations。
- `main.tsx` 删除 Settings 页面实现，只保留 route 绑定。
- `test:ui-contracts` 增加防线：`main.tsx` 不能再实现 Settings 工作台或包含 `settings-workbench-shell`。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 补充 `settingsPage.tsx` 模块边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/settings npm run render-check`

### 后续防线

- 页面级工作台继续从 `main.tsx` 拆出；`main.tsx` 只能保留 shell、route、轻量 page orchestration。
- Settings 新能力落在 `settingsPage.tsx` / `settingsControls.tsx` / `workspaceSettings.ts` 的对应层，不回流到主入口。

## 2026-05-25: Eval Bench 总览页面从 main.tsx 拆出

### 现象

“总览”已经承担总控工作台职责，包含 12 个粗粒度图表、运行态遥测、写入节奏和最近 run 摘要，但页面实现
和 helper 仍全部堆在 `main.tsx`。这会让入口文件继续膨胀，也削弱了后续对工作台图表、密度和动效做
页面级迭代的边界。

### 根因

总览页最初作为首页直接写入路由入口，后续完成高密度图表设计后只更新了视觉层，没有同步抽出页面模块。

### 影响范围

- 影响 Eval Bench dashboard 前端模块边界和 `main.tsx` 可维护性。
- 不改变总览指标语义、API 请求、排行榜主指标或后端存储结构。

### 修复方式

- 新增 `overviewPage.tsx`，承载总览页 query、图表装配、运行态遥测和最近 run 摘要。
- `main.tsx` 删除总览页面实现，只保留 route 绑定。
- `test:ui-contracts` 增加防线：`main.tsx` 不能再实现 `OverviewPage` 或包含总览工作台 class。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 补充 `overviewPage.tsx` 模块边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- 首页新增图表、遥测和最近 run 设计时只改 `overviewPage.tsx` 与样式层，不回流到 `main.tsx`。
- 页面级工作台继续按模块拆分；`main.tsx` 保持 shell、route 和轻量 orchestration。

## 2026-05-25: Eval Bench 基准集页面从 main.tsx 拆出

### 现象

`main.tsx` 仍承载基准集目录、创建副本弹窗、基准集真值检查器、样本列表和分页组件。这些逻辑已经属于
独立工作台页面，并且会持续迭代高级筛选、检查器滚动和样本导航；继续留在入口文件会让 route shell 与
业务页面互相缠绕。

### 根因

基准集页面早期为了接通 benchmark copy 和真值浏览器直接写入 `main.tsx`，后续只抽出了表格、viewer
和分栏基础组件，没有把页面容器和检查器编排拆出。

### 影响范围

- 影响 Eval Bench dashboard 前端模块边界、代码分块和样本检查器可维护性。
- 不改变 benchmark API、样本筛选语义、viewer 渲染语义或后端 benchmark copy 结构。

### 修复方式

- 新增 `benchmarksPage.tsx`，承载基准集目录、创建副本弹窗、基准集真值检查器和相关 helper。
- `/benchmarks` 与 `/benchmarks/$benchmarkId` 改为 lazy route，加载独立 `benchmarksPage` chunk。
- 新增 `samplePager.tsx`，将 benchmark/run 检查器共享分页按钮收敛到一个组件。
- `test:ui-contracts` 增加防线：`main.tsx` 不能再实现 Benchmarks 页面或 benchmark 检查器。
- layout smoke 增加 chunk 检查：`/benchmarks` 必须加载独立 `benchmarksPage` chunk。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 补充模块边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/benchmarks npm run render-check`

### 后续防线

- 基准集列表、创建副本、真值检查器和相关筛选只在 `benchmarksPage.tsx` 中演进，不回流到 `main.tsx`。
- 检查器分页继续复用 `samplePager.tsx`，避免 benchmark/run 两套分页按钮和滚动行为分叉。

## 2026-05-25: Eval Bench Runs 页面和共享样本 Viewer 从 main.tsx 拆出

### 现象

`main.tsx` 仍承载结果库高级检索、导入预测弹窗、run note 编辑器、run 样本检查器和完整样本 viewer。
这些能力对应目标中的 run note、agent 可导入预测、样本翻页、viewer 偏好状态和滚动布局，是后续高频迭代面；
继续留在入口文件会让 route shell、run 工作台和成对样本对比共用逻辑混在一起。

### 根因

Run Inspector 最初和成对样本对比共享同一个 `SampleViewer`，为了快速复用直接写在 `main.tsx`。
后续虽然抽出了 `viewerCanvas.tsx`、`viewerPanels.tsx`、`viewerMetrics.ts` 和 `samplePager.tsx`，
但没有把页面容器和共享 viewer 编排拆出。

### 影响范围

- 影响 Eval Bench dashboard 前端模块边界、代码分块、run note 编辑器和样本 viewer 可维护性。
- 不改变 run note API、导入预测 payload、样本筛选语义、viewer 偏好持久化或后端评估语义。

### 修复方式

- 新增 `runsPage.tsx`，承载结果库、导入预测弹窗、run note 编辑器、run 样本检查器和相关 helper。
- `/runs` 与 `/runs/$runId` 改为 lazy route，加载独立 `runsPage` chunk。
- 新增 `sampleViewer.tsx`，作为 Run Inspector 和成对样本对比共享叠图 / 对象检查器真源。
- `main.tsx` 删除 Runs 页面实现，只保留 shell、route 和成对样本对比的轻量页面容器。
- `test:ui-contracts` 增加防线：`main.tsx` 不能再实现 Runs 页面或 run 检查器。
- layout smoke 增加 chunk 检查：`/runs` 和 run inspector 必须加载独立 `runsPage` chunk。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 补充模块边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/runs npm run render-check`

### 后续防线

- 结果库、导入预测、run note、run 样本筛选和 run 检查器只在 `runsPage.tsx` 中演进。
- Run Inspector 与成对样本对比继续复用 `sampleViewer.tsx`，避免 viewer 偏好状态和对象检查器出现双轨实现。

## 2026-05-25: Eval Bench 成对样本对比详情从 main.tsx 拆出

### 现象

`main.tsx` 已经只剩 shell 和少量 route，但仍承载 `/compare/$baseline/$candidate/$sampleIndex` 的成对样本
对比详情、左右分栏和 SampleViewer 调用。这样会让 Compare 工作台的详情页继续成为入口文件里的业务页面。

### 根因

成对样本详情最初为了复用 run sample viewer 直接写在 `main.tsx`，直到 `sampleViewer.tsx` 抽出后才具备
干净迁移条件。

### 影响范围

- 影响 Eval Bench dashboard 前端模块边界和 compare sample 代码分块。
- 不改变 comparison report API、样本对比 URL、左右分栏持久化 key 或共享 viewer 行为。

### 修复方式

- 新增 `comparisonSamplePage.tsx`，承载成对样本对比详情、左右 run panel 和 comparison sample query。
- `/compare/$baselineRunId/$candidateRunId/$sampleIndex` 改为 lazy route，加载独立 `comparisonSamplePage` chunk。
- `comparisonSamplePage.tsx` 继续复用 `sampleViewer.tsx`，不维护第二套 GT / Prediction 叠图。
- `test:ui-contracts` 增加防线：`main.tsx` 不能再实现成对样本详情，详情页必须复用共享 `SampleViewer`。
- layout smoke 在存在 comparison artifact 时打开第 0 个样本对比详情，并检查独立 chunk 与 viewer 结构。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 补充模块边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/compare npm run render-check`

### 后续防线

- Compare 总览继续在 `comparePage.tsx` 中演进；成对样本详情只在 `comparisonSamplePage.tsx` 中演进。
- Run Inspector 与 Compare sample 继续共用 `sampleViewer.tsx`，避免叠图、对象检查器和偏好状态分叉。

## 2026-05-25: Eval Bench Overview signal deck 与 20+ 图表矩阵

### 现象

总览页已经有 12 个粗粒度 mini chart，但 ops 指标、写入节奏和实时遥测仍各占一行；用户继续要求总览拥有更多
可视化图表，并且进一步压缩信息密度。

### 根因

Overview 的上一轮设计虽然把最近 run 改成窄事件轨，但顶部运行态信号仍沿用多条独立 block。主矩阵也只覆盖
run / benchmark 的静态汇总，job、service、scheduler、备注覆盖和预测规模没有以图表形式进入总控视角。

### 影响范围

- 影响 Eval Bench dashboard 首页的信息架构、总览密度和实时控制台可读性。
- 不改变后端 API、run note 存储、rank-board 主指标或 eval metric 语义。
- 总览仍不展示 precision / recall / IoU 等细粒度模型能力指标。

### 修复方式

- 将 ops、12 桶写入节奏和 scheduler/job/service telemetry 合并为一条 `overview-signal-deck`。
- 总览主矩阵扩展到 20+ 个粗粒度 mini chart，新增 parser、viewer profile、预测规模、备注覆盖、job 状态、job 类型、service 状态、service 类型、实时信号和 scheduler 资源图表。
- 进一步压缩 chart card、ring、bar row 和 grid row 高度，避免少量数据时空白拉伸。
- compact 视口下如果图表矩阵需要更多垂直空间，由 `.overview-chart-matrix` 自己滚动，避免外层 `hidden` 裁切图表。
- 最近 run 从右侧整列改为底部横向事件条，释放主图表矩阵宽度，避免小数据量时出现大块空白侧栏。
- `layout-smoke-check.mjs` 将 Overview mini chart 下限提高到 20，并检查 signal deck 本身不能退化成大块低密度 block；同时检查图表矩阵需要滚动时不能 hidden 裁切。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- Overview 新增运行态内容必须优先进入 signal deck 或 mini chart 矩阵，不再新增整行低密度 block。
- 细粒度模型能力指标继续留在 Rank Board、Run Inspector 和 Compare，不进入总览。

## 2026-05-25: Eval Bench Agent 生命周期 CLI 补齐

### 现象

Dashboard API 已经支持 run archive/delete、job cancel/delete/log tail、service delete、backend log 和 scheduler
status，但 CLI 仍主要覆盖创建、列表、评估和 note。agent 如果要完成生命周期操作，仍容易退回到直接改
SQLite、移动 store 目录或猜 runtime log 路径。

### 根因

前期优先把人工 Dashboard 操作闭环做通，CLI 只补了 preflight、create、list、rank-board、run note 和
service 基础命令；API 侧新增的生命周期操作没有同步抽出共享后端 helper，也没有进入 CLI contract。

### 影响范围

- 影响 Eval Bench 的 agent 可操作性和排障路径。
- 不改变 job 状态机、service 管理语义、rank-board 排序或评估指标。
- 删除 run/service 仍走 trash，避免直接丢证据。

### 修复方式

- 新增 `log_utils.py`，集中维护 backend log tail、job runtime log tail 和 job log path 解析，Dashboard API 与 CLI 共用。
- `EvalBenchStore` 新增 `archive_run` / `delete_run`，Dashboard API 和 CLI 不再各自直接改 manifest 或移动 run 目录。
- CLI 新增 `dashboard-state`、`scheduler-status`、`backend-logs`、`archive-run`、`delete-run`、
  `cancel-job`、`delete-job`、`job-logs` 和 `delete-service`。
- 补充 CLI 测试，覆盖 state/log/run/job/service 生命周期命令。
- README、`docs/eval_bench_architecture.md` 和 `docs/scripts.md` 补充 agent CLI 边界和命令入口。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py`
- `python3 -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`

### 后续防线

- 新增 Dashboard 生命周期 API 时必须同步考虑 CLI 入口；agent 不应通过手写 SQLite、直接改 run manifest 或
  直接拼 runtime log 路径完成操作。
- run artifact 生命周期优先落到 `EvalBenchStore`；job record 生命周期优先落到 `EvalBenchDatabase`；
  service 生命周期优先落到 `EvalBenchServiceManager`。

## 2026-05-25: Eval Bench Overview 混合图表墙

### 现象

总览页已经有 20+ 个 mini chart，但所有图表都使用同一套 ring + bar 形态，最近 run 仍作为独立底部面板存在。
用户继续要求首页加入更多可视化图表，并进一步压缩信息密度。

### 根因

上一轮只提升了图表数量和 signal deck 密度，没有把图表形态、最近 run 布局和验收防线同步升级。
单一图表形态会让控制台看起来像重复卡片墙，最近 run 独立面板也会消耗一个横向区域。

### 影响范围

- 影响 Eval Bench dashboard Overview 的空间分配、图表密度和视觉扫描效率。
- 不改变后端 API、rank-board 主指标、评测指标语义或 run/job/service 状态机。
- Overview 继续只展示粗粒度运行态和数据规模信号，不展示 precision / recall / IoU 等细粒度指标。

### 修复方式

- 总览图表矩阵扩展为 40+ 个粗粒度 mini chart，补充 run benchmark、run task set、metric profile、模型来源、
  推理 backend、served model、TP/CUDA/batch/token/pixel/sampling 配置、prompt hash、report 规模、benchmark 新鲜度、
  benchmark 来源、job 阶段、job/service health、service 新鲜度和 scheduler loop 等信号。
- `OverviewMiniChartPanel` 支持 ring、rails、cells 和 meter 四种微图表形态，避免所有 tile 使用同一视觉算法。
- 最近 run 改为 `overview-recent-card`，嵌入 `.overview-chart-matrix`，只显示最新 4 条紧凑事件，不再占独立大块区域。
- 进一步压缩 console、signal deck、rhythm bar、chart tile 和 run row 高度，把剩余空间让给图表矩阵。
- `layout-smoke-check.mjs` 将 Overview mini chart 下限提高到 40，并检查四种图表形态和最近 run 矩阵嵌入关系。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- Overview 新增信号优先转换为现有微图表形态或新增受控 chart kind，不能回退到整行说明块或大面积列表。
- 最近 run 只作为矩阵内事件 tile 存在；如果需要更多条目，应通过 Runs 页承载，而不是扩大 Overview 面板。

## 2026-05-25: Eval Bench row/chip 交互原语收敛

### 现象

Run Inspector、Benchmark Inspector、Jobs 的 detection 子任务 label chip 和 Sample Viewer 工具 chip
都在业务页里直接写 raw `<button>`，并各自拼 `sample-row` / `query-chip` 的 selected/active class。

### 根因

前期优先收敛了弹窗和标准命令按钮，但样本行与 query chip 属于“带业务语义的按钮”，没有进入 `ui.tsx`
的共享组件层。后续页面如果继续复制 className 拼接，容易出现 selected、active、aria 语义不一致。

### 影响范围

- 影响 Eval Bench dashboard 的 Run Inspector、Benchmark Inspector、Jobs label 子任务和 Sample Viewer 工具 chip。
- 不改变样本分页、label 子任务策略、viewer 偏好状态或 API。

### 修复方式

- `ui.tsx` 新增 `SelectableRowButton`，统一维护 sample row 的 selected class 和 `aria-current`。
- `ui.tsx` 新增 `OptionChipButton`，统一维护 query/label chip 的 active class 和 `aria-pressed`。
- `runsPage.tsx` 与 `benchmarksPage.tsx` 的样本列表行改用 `SelectableRowButton`。
- `jobsPage.tsx` 的 detection label chips 和 `sampleViewer.tsx` 的检查器收起/展开 chip 改用 `OptionChipButton`。
- `test-ui-contracts.mjs` 增加静态防线，禁止这些高频 row/chip 调用点回退成 raw className 拼接。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- 新增样本行、候选项 chip 或 label chip 时先复用 `SelectableRowButton` / `OptionChipButton`。
- 画布 HUD、comparison delta card 这类具有独立视觉语义的复杂控件可以保留专用组件，但不要直接复制
  `sample-row` 或 `query-chip` class 拼接。

## 2026-05-25: Eval Bench target label preflight 防线

### 现象

Detection 子任务已经通过 `target_labels` 在 evaluator、store、worker、CLI、API 和前端 manifest 中贯通，
但 `preflight-job` 没有校验显式 label 是否存在于 benchmark label index。用户或 agent 拼错 label 时，
job 仍可能进入队列，最终得到低价值或误导性的评测结果。

### 根因

`job_spec.py` 的 preflight 只检查 benchmark 是否存在、task 是否支持、model/prompt/runtime 是否可解析；
target label scope 的真源在 `label_policy.py`，但 preflight 没有把 resolved payload 和 benchmark manifest
中的 `labels` 索引做交叉检查。前端 manifest 工具也只同步 `target_labels`，没有维护
`target_labels_source`，容易让显式选择、prompt 默认和清空默认策略的来源不够清晰。

### 影响范围

- 影响 agent CLI、Dashboard Jobs preflight 和入队前的 detection 子任务校验。
- 不改变 evaluator 的 label 过滤算法、run sample scope、rank-board 或 comparison 语义。

### 修复方式

- `preflight_job_payload()` 在 benchmark manifest 可读时校验 task 是否属于 benchmark tasks。
- `preflight_job_payload()` 在 benchmark label index 非空时拒绝未知 `target_labels`，并在 label index 缺失时给出 warning。
- CLI `create-job` 和 Dashboard `/api/jobs` 复用 preflight 后，将非阻塞 warning 写入 job metadata 的
  `preflight_warnings`，避免 agent 只在一次性 preflight 输出里看到风险。
- `manifestTools.ts` 应用 prompt template 时把 prompt metadata label 标记为 `target_labels_source=prompt_metadata`；
  用户手动修改 label 时标记为 `explicit`，清空 label 时同时清除 source，让后端重新走默认 policy。
- CLI、Dashboard API 和 manifest tools 均补充 focused 测试，覆盖 unknown label 拒绝和 warning 持久化。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py projects/eval_bench/tests/test_dashboard.py`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_worker.py`
- `cd projects/eval_bench/frontend && npm run test:manifest-tools`
- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `python3 -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`

### 后续防线

- 新增任务或 prompt template 时必须保证 benchmark label index、prompt metadata 和显式 `target_labels`
  的语义可由 preflight 验证。
- agent 创建 job 必须先跑 `preflight-job`；不能绕过 label 子任务校验直接写 SQLite job record。

## 2026-05-25: Eval Bench Overview 活动矩阵

### 现象

总览页已经有 40+ 个粗粒度 mini chart，但 signal deck 中的时间图仍只表达 run 写入节奏。用户继续要求
总览拥有更多可视化图表，并进一步压缩信息密度。

### 根因

原来的 12 桶 rhythm 只消费 `runs.created_at`，没有把 job queue 和 service runtime 的变化纳入同一时间面板。
这会让实时控制台看起来仍偏静态，agent 或用户需要跨多个区域才能判断 run、job、service 是否同步活跃。

### 影响范围

- 影响 Dashboard Overview 的 signal deck 视觉和 layout smoke 验收。
- 不改变后端 store、job、service 或 scheduler API；前端只消费已有 `runs`、`jobs`、`services` 字段。
- Overview 仍不展示 precision / recall / IoU 等细粒度评测指标。

### 修复方式

- 将原单条 `OverviewWriteRhythm` 升级为 `OverviewActivityMatrix`，用 Run / Job / Service 三条泳道展示最近
  12 个日期桶的活动密度。
- 活动矩阵继续留在 signal deck 内，与 ops 和实时遥测同屏，不新增整行低密度面板。
- 复用同一套日期桶生成 helper，避免 run timeline 和 activity matrix 各自维护日期桶算法。
- `layout-smoke-check.mjs` 将 Overview 时间图验收更新为 3 条 12-cell 活动泳道。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run render-check`

### 后续防线

- Overview 新增实时信号优先进入活动矩阵、telemetry cell 或 mini chart，不新增大块说明面板。
- 如果后续需要分钟级实时状态，应新增专门的轻量 API 聚合，不要在前端扫描原始 artifact 文件。

## 2026-05-25: Eval Bench Viewer 控件原语收敛

### 现象

Viewer 控制面板的图层预设仍直接手写 `compact-select` 外壳，标签筛选 chip 也直接写 raw button 和
`label-select active` class 拼接。这个面板是 Run Inspector、Benchmark Inspector、Compare 和 Settings
预览共享的核心交互，后续容易再次出现按钮、select 和 chip 风格漂移。

### 根因

前面已经收敛了弹窗、标准动作按钮、样本行和部分 query chip，但 viewer panel 比较早抽出，保留了局部控件
实现。`test:ui-contracts` 也只覆盖了 sample viewer 工具 chip，没有覆盖 viewer panel 内的图层预设和
label chip。

### 影响范围

- 影响 Eval Bench Dashboard 的 viewer 控制面板组件边界和 UI 一致性。
- 不改变 viewer 图层状态、label 过滤语义、样本翻页或后端 API。

### 修复方式

- Viewer 图层预设改用 `CompactSelectControl`。
- Viewer label chip 改用 `OptionChipButton`，保留 `label-select` 专用视觉 class。
- `test-ui-contracts.mjs` 增加静态防线，防止 viewer panel 回退到手写 compact select 或 raw label-select button。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录控件原语边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 新增 viewer 控件先判断是否可复用 `controlPrimitives.tsx` 或 `ui.tsx`；只有 object row、canvas HUD
  这类有独立复杂交互语义的控件保留专用实现。

## 2026-05-25: Eval Bench 表单提交按钮双轨收口

### 现象

Benchmark 创建、prediction 导入和 service 登记弹窗的提交动作已经使用 `ActionButton`，但仍额外挂着
`form-submit-button` 页面私有 class。该 class 当前没有独立样式价值，却会让后续开发者误以为表单提交
需要另一套按钮样式入口。

### 根因

早期弹窗表单先用页面局部 class 接入视觉，后续把按钮迁移到 `ActionButton` 时没有删除旧 class，
`test:ui-contracts` 也没有覆盖弹窗表单 submit 的回流风险。

### 影响范围

- 影响 Dashboard 的 Benchmark、Runs import、Services 三个弹窗表单。
- 不改变 API、表单字段、提交 mutation 或弹窗滚动行为。

### 修复方式

- 删除业务页里 `ActionButton` 上的 `form-submit-button` 附加 class。
- 删除 `design.css` 中已无调用点的 `.form-submit-button` 样式，并更新图标设计文档中的按钮规则。
- `test-ui-contracts.mjs` 增加 `assertNoLegacyFormSubmitClass()`，覆盖 Jobs、Benchmarks、Runs 和 Services。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录 submit 也属于标准 `ActionButton` 边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 新增弹窗表单提交动作直接使用 `ActionButton variant="primary"`，不要再引入页面私有 submit class。

## 2026-05-25: Eval Bench 局部 select 原语收口

### 现象

Runs 导入预测弹窗、Services 登记弹窗、Compare 左侧 run 选择轨和 Settings label 颜色角色输入仍在业务页
或分组控件里直接手写 `<select>`。
这些位置已经完成弹窗和按钮组件化，但局部 select 还没有进入统一 primitive，后续容易继续复制不同
label/select 外壳。

### 根因

之前只为 manifest toolbar、viewer 和 settings 抽出了 `CompactSelectControl`，它适合紧凑工具条，
但不适合弹窗表单和选择轨这种普通 form label。业务页因此继续保留 raw select，`test:ui-contracts`
也没有把这些页面纳入 select 回流防线。

### 影响范围

- 影响 Eval Bench Dashboard 的 Runs import、Services registration、Compare run rail 和 Settings
  label color editor。
- 不改变筛选语义、表单字段、提交 mutation、对比报告加载或后端 API。

### 修复方式

- 在 `controlPrimitives.tsx` 新增 `FormSelectControl`，统一普通 form/rail select 的 label、title、
  disabled、required 和 option 渲染。
- Runs 导入预测弹窗的 benchmark/task select 改用 `FormSelectControl`。
- Services 登记弹窗的 service kind select 改用 `FormSelectControl`。
- Compare 左侧基线/候选 run 选择轨改用 `FormSelectControl`。
- Settings label 颜色角色输入改用隐藏 label 的 `FormSelectControl`，保留紧凑 inline 栅格。
- `test-ui-contracts.mjs` 增加 `assertNoRawSelectElement()`，覆盖 SettingsControls、Runs、Services
  和 Compare，阻止这些页面重新手写 raw `<select>`。
- README、`docs/scripts.md`、`docs/eval_bench_architecture.md` 和
  `docs/eval_bench_ui_icon_design.md` 同步记录 select primitive 边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 页面级高级检索继续走 `filterControls.tsx`；普通弹窗表单和选择轨 select 必须走
  `FormSelectControl`，紧凑 toolbar select 必须走 `CompactSelectControl`。

## 2026-05-25: Eval Bench 检查器过滤空结果不能卸载工作区

### 现象

Benchmark Inspector 和 Run Inspector 的样本过滤如果返回 0 条样本，页面外层直接渲染全页
`EmptyState`。这样高级检索按钮、样本列表和主画布都被卸载，用户无法在原检查器工作区里撤销过滤，
表现为过滤后“卡住”。

### 根因

检查器把 `samples.length === 0` 同时当成“基准集/run 没有样本”和“当前过滤没有命中”。引入
AdvancedFilterBar 后，过滤空结果是正常工作流，不能继续用全页空状态处理。

### 影响范围

- 影响 Benchmark Inspector 的 label 过滤和 Run Inspector 的 error/label 过滤。
- 不改变后端分页 API、样本详情 API、过滤参数语义或 viewer 叠图渲染。

### 修复方式

- Benchmark / Run 检查器只在无活动过滤且 `page.total === 0` 时使用全页 EmptyState。
- 有活动过滤时保留 `ResizableSplit`、`inspector-sidebar`、`AdvancedFilterBar` 和 `viewer-panel`，
  只在样本列表和主画布区域显示“没有符合过滤条件的样本”。
- 分页 offset 超过新 total 时通过 `sampleNavigation.ts` 的共享 helper 自动夹回最后一页，避免过滤或数据变化后停在不可达页。
- `layout-smoke-check.mjs` 增加过滤空结果模拟：拦截 benchmark/run samples API 返回 0 命中，
  验证过滤入口、样本列表和主画布空状态仍留在同一个 inspector split 内。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录检查器过滤空结果边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 新增检查器过滤条件时，过滤空结果必须保留原工作区和撤销入口；不能用全页 EmptyState 替代局部空结果。

## 2026-05-25: Eval Bench 高级检索统一清空动作

### 现象

各页面已经把筛选表单收敛到 `AdvancedFilterBar`，但用户筛选多项条件后仍需要逐项展开并手动恢复。
Rank Board 里排序字段和排序方向这类没有 `all` 选项的 select 还会被计入“生效条件”，导致默认状态也显示
已有过滤条件。

### 根因

`AdvancedFilterBar` 只统一了折叠外壳和控件渲染，没有统一“默认值”和“清空”语义。生效条件计数只判断
非空且不等于 `all`，无法区分筛选 select 和排序 select。

### 影响范围

- 影响 Benchmarks、Jobs、Services、Runs、Compare、Rank Board 以及样本检查器的高级检索交互。
- 不改变任何后端查询参数、API filter 语义或 rank-board 排序结果。

### 修复方式

- `AdvancedFilterBar` 新增统一清空动作，只有存在生效条件时显示。
- search/number 的默认值是空字符串；包含 `all` 的 select 默认回到 `all`；没有 `all` 的 select
  默认回到第一个选项，用于排序字段、排序方向等非过滤控件。
- 生效条件计数改为和同一份 `defaultFilterValue()` 对比，避免 Rank Board 默认排序被误算成过滤条件。
- `test-ui-contracts.mjs` 增加静态防线，要求清空动作和默认值判定留在 `filterControls.tsx`。
- `layout-smoke-check.mjs` 在 Runs 和 Rank Board 中实际输入检索词、点击清空，并验证输入、清空按钮和摘要恢复默认状态。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录高级检索清空边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 新增页面级 filter 时继续复用 `AdvancedFilterBar`，不要在页面里单独实现清空按钮或生效条件计数。

## 2026-05-25: Eval Bench Overview 高密度图表墙再压缩

### 现象

总览页已经有 40+ 个粗粒度 mini chart，但用户继续要求首页拥有更多可视化图表，并明确要求进一步压缩
信息密度。现有图表墙里时间类数据仍偏少，run/job/service 之外的 artifact 覆盖、样本权重和配置快照也
没有形成足够多的可扫描图表。

### 根因

总览页前几轮收口主要解决“删掉低价值大块区域”和“把最近 run 嵌入矩阵”，但图表形态仍集中在 ring、
rails、cells 和 meter 四类。时间序列只能以三泳道活动矩阵表达，无法在主矩阵中继续压缩展示 run/job/service
日历、小时段、weekday、样本权重和 artifact 覆盖这些粗粒度运营信号。

### 影响范围

- 影响 Eval Bench Dashboard 首页的信息架构、图表密度和 layout smoke 验收。
- 不改变后端 API、store schema、排行榜主指标、评测语义或 run/job/service 状态写入。
- Overview 仍不展示 precision / recall / IoU 等细粒度评测指标。

### 修复方式

- `overviewPage.tsx` 新增 `spark` 和 `mosaic` 两种 mini chart 形态。
- 主矩阵补充 run/job/service 日历、run/job 小时段、weekday、artifact 覆盖、配置快照、样本按 task/label/layer/split 权重、
  prediction/report 按状态权重、benchmark manifest 覆盖和 task/label span 等粗粒度图表。
- CSS 进一步压缩 overview console、signal deck、chart card、最近 run tile、bar list、ring、rail、cell 和 telemetry
  的固定高度；最近 run 在宽屏下变成 4 列紧凑事件 tile。
- `layout-smoke-check.mjs` 将 Overview mini chart 下限提高到 56，并要求同时出现 ring / rails / cells / meter /
  spark / mosaic 六种形态。
- `test-ui-contracts.mjs` 增加静态防线，要求 `OverviewSparkChart`、`OverviewMosaicChart` 和 56+ chart spec 留在
  `overviewPage.tsx`。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步更新 Overview 图表墙边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ SCREENSHOT_PATH=/home/tanjingyuan/code/arrow-vlm/temp/eval_bench_overview_dense.png npm run render-check`
- `git diff --check`

### 后续防线

- Overview 新增内容优先进入已有 mini chart 类型或新增可复用微图表类型；不要恢复长列表、说明 block 或大面积低密度 card。
- 细粒度模型能力指标继续留在 Rank Board、Run Inspector 和 Compare，不进入总览。

## 2026-05-25: Eval Bench comparison 详情补齐 agent CLI

### 现象

Agent 已经可以用 `compare-runs` 生成 comparison，也可以用 `list-comparisons` 查询历史列表，但缺少稳定 CLI
读取某一份已保存 comparison 的完整 report，也缺少 CLI 读取成对样本详情。排障时 agent 仍需要重跑
`compare-runs`，或者直接读取 `eval_bench_store/exports/comparisons` 和 run artifact 文件。

### 根因

Comparison 模块只暴露了“生成”和“列表摘要”两个稳定入口；Dashboard 的成对样本详情 payload 还留在
dashboard route 私有 helper 中。CLI/API/前端没有共享一个读取 comparison report 和样本详情的后端语义。

### 影响范围

- 影响 Eval Bench 的 agent 操作面：comparison 详情和样本详情读取不够稳定。
- 不改变 comparison report 格式、dashboard route、排行榜、评测指标或前端页面结构。

### 修复方式

- `comparison.py` 新增 `comparison_id_for_runs()`、`load_comparison_report()`、
  `comparison_sample_detail_payload()` 和 `run_sample_detail_payload()`。
- `compare-runs` 复用 `comparison_id_for_runs()`，避免 comparison id 生成逻辑继续散落。
- 新增 CLI `show-comparison`，支持通过 `--comparison-id` 或 baseline/candidate run id 读取已保存 report。
- 新增 CLI `show-comparison-sample`，通过 `EvalBenchStore.run_sample_detail()` 读取 baseline/candidate
  同一 sample index 的 scoped GT/prediction/diagnostics。
- Dashboard route 的 run sample payload 改为复用 `comparison.py` 的 `run_sample_detail_payload()`，
  只在 API 层补 image URL。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录 agent comparison CLI 边界。

### 回归测试

- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`
- `uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run pytest -q projects/eval_bench/tests/test_dashboard.py`
- `git diff --check`

### 后续防线

- Agent 新增读取对象时，先提供 CLI/API 稳定入口；不要要求 agent 读取前端状态、SQLite 或 store artifact 文件。
- Dashboard route 私有 payload helper 如果开始被 CLI 需要，应收敛到后端模块真源，而不是复制一份。

## 2026-05-25: Eval Bench Rank Board 主指标切换语义修正

### 现象

Rank Board 的 API/CLI/前端已经允许 `sort_by=recall_iou50`、`precision_iou50`、`mean_iou` 等排序，
但返回的 `primary_metric`、`primary_metric_label` 和 `score_formula` 仍固定为 F1。前端切到
Recall 排序后，页面标题和 chip 仍显示“按 F1”，agent 读取 JSON 时也无法区分“当前排行主指标”和
“默认 F1”。

### 根因

前一轮只把“默认不加权、默认 F1”收口了，但没有把非加权主指标切换建模为后端返回语义。`score`
仍被固定当作 F1 兼容字段，导致 `sort_by` 已变化时，主指标字段和 entry 级 score 不同步。

### 影响范围

- 影响 Eval Bench Rank Board 的 API/CLI 返回语义和 Dashboard 排序说明。
- 不改变默认 F1 排行结果、显式 weighted scheme、comparison report 或 evaluator 指标计算。

### 修复方式

- `EvalBenchStore.rank_board()` 在非加权模式下根据 `sort_by` 推导 `primary_metric` 和
  `primary_metric_label`；默认仍是 `f1_iou50`。
- 非加权 entry `score` 改为镜像当前主指标；当 `sort_by` 是 `created_at` 或 `run_id` 这类列表排序维度时，
  主指标和 score 仍回落到 F1。
- Rank Board 前端标题和 formula chip 改为同时显示主指标与排序维度，避免主指标和列表排序混淆。
- README 和 `docs/eval_bench_architecture.md` 同步更新 Rank Board 主指标切换边界。

### 回归测试

- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`
- `uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run pytest -q projects/eval_bench/tests/test_dashboard.py`
- `cd projects/eval_bench/frontend && npm run build`
- `git diff --check`

### 后续防线

- Rank Board 新增可排序指标时，必须同时补齐 `primary_metric`、`primary_metric_label`、entry `score`
  和前端标签展示测试。
- `created_at`、`run_id` 这类列表排序维度不能伪装成评测主指标。

## 2026-05-25: Eval Bench target label scope 增加 agent 查询入口

### 现象

Detection label 子任务已经在 job manifest、preflight、evaluator、run sample scope 和前端 chip 里落地，
但 agent 在创建 job 或导入 prediction 前，仍需要自己组合 `list-benchmarks`、`list-prompt-templates`
和 `label_policy.py` 规则，才能知道最终会评哪些 `target_labels`、候选 label 是什么、是否拼错。

### 根因

后端有 `label_policy.py` 作为默认 label scope 真源，store 也能给出 benchmark label summary，
prompt registry 能给出 prompt metadata，但缺少一个 agent-safe 的聚合查询 CLI。label 子任务的解释链路
对 UI 是清楚的，对 agent 仍不够一等。

### 影响范围

- 影响 Eval Bench agent 操作面：创建 detection label 子任务前的校验和发现能力。
- 不改变 evaluator 匹配算法、job preflight 失败条件、run manifest 格式、Dashboard 前端交互或已有 report。

### 修复方式

- `label_policy.py` 新增 `target_label_resolution_payload()`，统一返回最终 target labels、来源、candidate labels、
  benchmark/prompt/explicit 三类输入、valid/errors/warnings 和 label 子任务支持状态。
- CLI 新增 `resolve-target-labels`，从 `EvalBenchStore.benchmarks()` 和 prompt template registry 读取输入，
  再复用 `label_policy.py` 生成 agent 可消费 JSON。
- `test_cli.py` 覆盖 prompt metadata 默认范围和未知 label 拼写错误。
- README、`docs/scripts.md` 和 `docs/eval_bench_architecture.md` 同步记录 agent 查询边界。

### 回归测试

- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`
- `uv run pytest -q projects/eval_bench/tests/test_cli.py`
- `uv run pytest -q projects/eval_bench/tests/test_job_spec.py`
- `uv run pytest -q projects/eval_bench/tests/test_evaluator.py`
- `uv run pytest -q projects/eval_bench/tests/test_prediction_import.py`
- `git diff --check`

### 后续防线

- 新增 label scope 来源时，必须先扩展 `label_policy.py` 的 resolution payload，再暴露到 CLI/API/UI。
- Agent 不应直接扫描 benchmark artifact、SQLite prompt table 或前端 manifest state 来推断 label 子任务。

## 2026-05-25: Eval Bench Compare label delta card 按钮原语收敛

### 现象

Compare 页的 label delta 卡片仍直接在业务页写 raw `<button className="label-delta-card ...">`，
并自己拼 active class。此前已经收敛了样本行、query chip、viewer label chip 和标准动作按钮，
但 Compare 的可选卡片没有进入公共按钮原语，后续容易继续出现 aria/active 语义双轨。

### 根因

`ui.tsx` 已有 `SelectableRowButton` 和 `OptionChipButton`，但它们分别服务行选择和紧凑 chip；
Compare label delta 是卡片式选项，早期没有对应共享组件，所以业务页保留了局部 raw button。

### 影响范围

- 影响 Eval Bench Dashboard Compare 页的 label delta 选择卡片。
- 不改变 comparison API、label delta 计算、筛选语义、样本跳转或布局结构。

### 修复方式

- `ui.tsx` 新增 `SelectableCardButton`，统一维护 card option 的 active class 和 `aria-pressed`。
- `comparePage.tsx` 的 `ComparisonLabelDeltaStrip` 改用 `SelectableCardButton`，保留原有
  `label-delta-card` 视觉 class。
- `test-ui-contracts.mjs` 增加静态防线，禁止 Compare label delta 卡片回退为 raw button。
- README 和 `docs/eval_bench_architecture.md` 同步记录可选卡片按钮原语。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `git diff --check`

### 后续防线

- 新增卡片式选项按钮时优先复用 `SelectableCardButton`；只有画布 HUD、对象行或快捷键捕获这类专用输入控件允许保留局部 raw button。

## 2026-05-25: Eval Bench Overview 高价值面板收口

### 现象

总览页从 56+ mini chart 图表墙降密度后，仍保留了 parser、配置快照、artifact 覆盖、备注新鲜度、推理参数桶等低频排障信息。页面可读性改善了，但面板价值没有被重新筛选，导致总控工作台仍像“把所有字段可视化”的集合。

### 根因

前一轮主要解决图表过密和纯 grid 问题，没有把“是否帮助调度判断”作为总览准入标准。旧图表 spec 裁剪后仍沿用了过多 run manifest / artifact 细节计算，和 Runs、Inspector、Rank Board 的职责重叠。

### 影响范围

- 影响 Eval Bench Dashboard 总览页的信息架构和第一屏判断效率。
- 不改变后端 API、rank board 排序语义、run note、sample viewer 或评估指标计算。

### 修复方式

- Overview 主区域收敛为 8-16 个高价值粗粒度面板，只保留运行态、评估覆盖、数据规模、label footprint、模型分布、job/service/scheduler 和近期写入信号。
- 删除总览里 parser、配置快照、artifact 明细、备注新鲜度、推理参数桶等低频排障面板及对应死计算。
- UI 合约和 layout smoke 从“24-40 个可读 chart”改为“8-16 个高价值 chart”，继续要求 2 列起、最多 4 列 capped masonry columns，且不能退回纯 grid。
- README、`docs/eval_bench_architecture.md` 和 `docs/scripts.md` 同步新的总览准入边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`
- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_dashboard.py -k 'rank_board'`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ npm run test:layout`
- `cd projects/eval_bench/frontend && EVAL_BENCH_URL=http://127.0.0.1:8766/ SCREENSHOT_PATH=/home/tanjingyuan/code/arrow-vlm/temp/eval_bench_overview_readable.png npm run render-check`

### 后续防线

- Overview 新增面板必须先回答是否帮助调度、覆盖和近期写入判断；低频复现线索和排障细节默认进入 Runs / Inspector / Rank Board。
- UI smoke 不再奖励更多图表数量，防线应限制上限并检查可读性、分栏上限和滚动边界。

## 2026-05-25: Eval Bench agent 单对象详情 CLI 补齐

### 现象

Agent 已能分页列出 benchmark、run、service 和 comparison，也能读取 run、sample 和 comparison
详情；但 benchmark summary 与 model service 详情缺少对应单对象 CLI。排障或自动编排时，agent
仍可能退回到扫描 benchmark artifact、读取 SQLite 或依赖 dashboard 前端 state。

### 根因

前几轮优先补齐列表、run/report/sample/comparison 和 lifecycle 命令，单对象读取能力没有按对象类型
完整对齐。`EvalBenchStore` 和 `EvalBenchServiceManager` 也缺少公开的单对象读方法，导致 CLI 没有
稳定复用点。

### 影响范围

- 影响 Eval Bench agent 操作面：benchmark summary 和 service 详情读取不够稳定。
- 不改变 dashboard API、store artifact 格式、service registry 表结构、run 评估或排行语义。

### 修复方式

- `EvalBenchStore` 增加 `benchmark(benchmark_id)`，作为 benchmark summary 单对象读取真源。
- `EvalBenchServiceManager` 增加 `service(service_id)`，复用现有 registry 与 runtime refresh 逻辑。
- CLI 增加 `show-benchmark` 和 `show-service`，输出 agent 可消费 JSON。
- README、`docs/eval_bench_architecture.md` 和 `docs/scripts.md` 同步记录 agent 读取边界。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py -k 'agent_filters or services_with_agent_filters'`
- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`
- `git diff --check`

### 后续防线

- 新增 dashboard 可操作对象时，必须同时检查列表 CLI 和单对象详情 CLI；agent 不应直接读取前端状态、
  SQLite 或 artifact 目录来补齐对象详情。

## 2026-05-25: Eval Bench 业务页按钮原语继续收口

### 现象

Dashboard 已经把主要动作按钮、样本行、label chip、Compare label delta card 和弹窗提交按钮收敛到公共
组件，但侧栏折叠、Settings 分组切换、Viewer 画布复位和 object row 仍在业务页直接写 raw `<button>`。
这些控件虽然功能正确，但 active、aria、dense/icon 语义继续散落在页面里，后续容易回退成各页独立样式。

### 根因

前几轮组件化优先处理高频表格、弹窗和筛选控件，没有把 shell 控件、settings 局部分组和 viewer object row
纳入同一套 button primitive 合约。静态测试也只覆盖了一部分已收敛控件。

### 影响范围

- 影响 Eval Bench Dashboard 前端组件边界和按钮风格一致性。
- 不改变 API、评估语义、viewer 几何计算、快捷键 registry、run note 或 rank board 数据流。

### 修复方式

- 侧栏折叠按钮改用 `IconActionButton`。
- Settings 分组切换和 viewer object row 改用 `SelectableCardButton`，统一 active/aria 语义。
- Viewer 画布复位按钮改用 `ActionButton`，保留 HUD 内部紧凑显示。
- `test:ui-contracts` 增加防线，禁止这些控件回退为业务页 raw button。
- README、`docs/eval_bench_architecture.md` 和 `docs/scripts.md` 同步更新按钮原语边界。

### 回归测试

- `cd projects/eval_bench/frontend && npm run build`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`

### 后续防线

- 新增业务页按钮时先判断是否是标准动作、图标动作、行选择、chip 或 card 选择；除极低层 canvas
  pointer/hud 控件外，不在页面里直接拼 raw button class。

## 2026-05-25: Eval Bench agent show CLI 覆盖 job 与模板对象

### 现象

Agent 已经可以列表查询 job templates、prompt templates 和 jobs，也可以创建 job、维护 prompt template；
但读取单个 job template、prompt template 或 job 详情时仍缺少稳定 CLI。自动编排或排障时容易退回到
直接读 SQLite、复用 list 输出中的局部字段，或者扫描 dashboard/store 状态。

### 根因

前几轮优先补齐了 create/list/preflight/lifecycle 和 benchmark/service 的单对象读取，模板对象和 job
record 没有同步形成 list + show 闭环。虽然数据库层已有 `get_job()` 和 `get_prompt_template()`，
CLI 没有暴露这些真源。

### 影响范围

- 影响 Eval Bench agent 操作面：job template、prompt template 和 job record 详情读取不够稳定。
- 不改变 job 状态机、prompt template registry schema、preflight 语义、dashboard API 或评估运行链路。

### 修复方式

- CLI 增加 `show-job-template`，从 `job_spec.job_templates()` 读取单个 manifest-first 模板。
- CLI 增加 `show-prompt-template`，复用 `EvalBenchDatabase.get_prompt_template()`。
- CLI 增加 `show-job`，复用 `EvalBenchDatabase.get_job()`。
- README、`docs/eval_bench_architecture.md` 和 `docs/scripts.md` 同步记录 agent list/show 边界。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py -k 'templates_for_agents or manifest_first_job'`
- `uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests`

### 后续防线

- 新增 agent 可操作对象时，必须同时补 list 和 show 的稳定 CLI；不允许让 agent 从 SQLite、前端 state
  或 list payload 中临时拼对象详情。

## 2026-05-25: Eval Bench label 子任务边界增加 keypoint 防线

### 现象

Detection 的 label 子任务已经贯通后端 policy、preflight、UI manifest 和 agent CLI；Jobs 页也只在
`task=detection` 时渲染 label 子任务面板。但这个 detection-only 边界缺少直接测试，后续改 UI 或
agent payload 时可能把 keypoint 误暴露为可任意选择 label 子集的任务。

### 根因

现有测试重点覆盖 detection 显式 label、拼写校验、prompt metadata 和 evaluator 过滤；keypoint 默认
arrow scope 有测试，但没有把 `label_subtasks_supported=false` 和 UI 面板 detection-only 作为合约。

### 影响范围

- 影响 Eval Bench label 子任务语义：detection 可选 label 子集，keypoint 只保留 arrow 关键点评估范围。
- 不改变 evaluator、worker、preflight、prompt template 或 run manifest 格式。

### 修复方式

- `test_cli.py` 补充 keypoint 的 `resolve-target-labels` 断言，确认返回 arrow 且
  `label_subtasks_supported=false`。
- `test:ui-contracts` 增加静态防线，要求 Jobs 页的 `LabelSubtaskPanel` 保持 `task !== "detection"`
  时返回 `null`。
- README 和 `docs/eval_bench_architecture.md` 明确 keypoint 不暴露 label 子任务 UI。

### 回归测试

- `PYTHONPATH=projects/eval_bench uv run pytest -q projects/eval_bench/tests/test_cli.py::test_cli_resolves_target_labels_for_agent_label_subtasks`
- `cd projects/eval_bench/frontend && npm run test:ui-contracts`

### 后续防线

- 新增任务类型或 label scope 策略时，必须同时更新 agent resolution payload 和 Jobs 页 UI 合约；
  不能只靠 prompt id 或页面默认值推断是否允许 label 子任务。
