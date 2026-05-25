# Shaft Eval Bench

Shaft Eval Bench 是面向 Shaft 视觉结构任务的内部评测工作台。它作为 Shaft 仓库内的子项目维护，依赖统一由仓库根目录 `pyproject.toml` 管理；子项目内不再维护第二份 `pyproject.toml`。

这个项目以 dashboard 为主，但核心边界保持清楚：

```text
benchmark GT + model inference -> raw-data-like prediction snapshot
prediction snapshot + eval spec -> metrics, previews, reports, comparisons, dashboard
```

这个拆分很重要：Shaft 的任务定义、prompt、codec 和 metric 仍在演进。已经完成的推理 run 应该可以在不重新跑模型的情况下，后续反复重新打分和重新可视化。

Eval Bench 不复用现有 online eval 数据流。它自己管理 benchmark 数据：从 `raw_data` 中选择验证 split，复制到 `eval_bench_store/benchmarks/`，后续 run 只消费这份不可变 benchmark copy。

## Dashboard 技术栈

Dashboard 是一个常驻的小型内部系统，不是 notebook 或一次性报告：

- 后端/API：`projects/eval_bench/eval_bench/dashboard.py` 中的 FastAPI
- 持久化 job registry：`eval_bench_store/db/eval_bench.sqlite` 中的 SQLite
- 持久化 model service registry：同一个 SQLite store，服务日志位于 `eval_bench_store/services/`
- 持久化 prompt template registry：同一个 SQLite store，供 job 创建、preflight 和 worker 解析共享
- Dashboard 后端日志：`eval_bench_store/logs/backend.log`，API 可通过 `/api/logs/backend` 读取 tail
- 前端：`projects/eval_bench/frontend` 中的 React + TypeScript + Vite
- 路由：TanStack Router
- 服务端状态：TanStack Query
- 表格：TanStack Table
- UI 基础组件：仓库内 `ui.tsx` 和 `controlPrimitives.tsx`
- 前端 metric 中间层：`projects/eval_bench/frontend/src/viewerMetrics.ts`
- 业务图标库：`projects/eval_bench/frontend/public/icons/eval-bench/*.png`，统一通过
  `projects/eval_bench/frontend/src/iconLibrary.tsx` 引用

FastAPI app 同时提供 `/api/*` 和构建后的 Vite SPA。本地前端开发时，Vite 会把 `/api` 代理到 dashboard API。

前端模块边界按功能拆分，不允许把新能力继续堆回 `main.tsx`：

- `main.tsx`：路由、页面装配和少量页面级 orchestration；不承载可复用业务规则。
- `api.ts`：dashboard API schema 与请求函数。
- `dashboardState.ts`：dashboard state query 的单一 hook 入口。
- `statusModel.ts`：job、run、service 的状态文案、tone、live 标记和操作权限。
- `workspaceSettings.ts`：浏览器本地设置 schema、normalize、快捷键 action registry 和 hooks。
- `sample_paths.py`：GT JSON 到原图路径、原图路径到 prediction JSON 路径的后端单一真源。
- `sample_scope.py`：Run sample 按 `target_labels` 裁剪实例、payload 和 diagnostics 的后端单一真源。
- `log_utils.py`：backend log、job runtime log tail 和 job log path 解析的后端共享工具，供 API 与 CLI 共用。
- `workspaceLayout.tsx`：workspace split pane、拖拽 resize 和尺寸持久化。
- `overviewPage.tsx`：总控工作台页面；承载下一步动作、评测管线、运行压力、活动矩阵、readiness switchboard 和最近 run 摘要。
- `benchmarksPage.tsx`：基准集目录、创建副本弹窗和基准集真值检查器；按懒加载路由拆分。
- `samplePager.tsx`：benchmark/run 检查器共享样本分页控件；分页按钮统一走 `ActionButton`。
- `runsPage.tsx`：结果库、导入预测弹窗、run note 编辑器和 run 样本检查器；按懒加载路由拆分。
- `sampleViewer.tsx`：Run Inspector 与成对样本对比共享的 GT / Prediction 叠图和对象检查器。
- `jobsPage.tsx`：评测中心、job queue、job create manifest 和 runtime log 组件。
- `runTables.tsx`：benchmark/run 表格、run 操作和 run 过滤。
- `rankBoardPage.tsx`：独立排行榜工作台；按懒加载路由拆分，承载 rank board 高级检索、facet 和排序 UI。
- `comparePage.tsx`：成对 run 对比工作台；按懒加载路由拆分，只承载 comparison 报告、历史对比和对比上下文。
- `comparisonSamplePage.tsx`：成对样本对比详情；按懒加载路由拆分，复用 `sampleViewer.tsx`。
- `settingsPage.tsx`：工作台设置页面；承载设置页查询、预览、分组装配和本地偏好 mutations。
- `filterControls.tsx`：跨页面复用的过滤下拉控件。
- `viewerCanvas.tsx`：图像 stage、pan/zoom、SVG instance layer 和坐标投影渲染。
- `viewerPanels.tsx`：viewer 的控制条、可见指标、对象列表和实例统计。
- `viewerGeometry.ts`：bbox、points、方向箭头、pan/zoom 边界和颜色解析等纯计算。
- `viewerMetrics.ts`：可见对象、TP/FP/FN、per-label metrics 和对象诊断中间层。
- `settingsControls.tsx`：设置页 section、preference row、label 颜色添加和快捷键编辑器。
- `servicesPage.tsx`：模型服务页的查询、mutations、日志和表单。
- `manifestTools.ts`：job manifest、prompt template 和 benchmark 默认值转换。
- `sampleNavigation.ts`：样本 URL query、分页 offset、样本 offset 合法化和相邻样本导航。
- `samplePager.tsx`：样本分页、目录分页控件和通用 offset clamp；Runs / Benchmarks / Jobs /
  Services / Compare / Rank Board 不各自手写 pager。
- `formatters.ts`：展示格式化、链接构造和输入目标判断。
- `controlPrimitives.tsx`：设置页、viewer、manifest toolbar、弹窗表单和对比选择轨共用的 number input、color input、select、toggle 基础控件。
- `iconLibrary.tsx`：Eval Bench 业务语义 PNG 图标的唯一路径映射；基础工具动作仍使用矢量图标。

## Dashboard 页面约定

- `评测中心` 是日常入口：同页展示短生命周期 job 队列、调度状态、runtime log 和最近 run 结果。
  job 成功后会在详情里直接链接到对应 run；完整历史仍可在 `结果库` 中查询、删除、归档和重新评估。
- `结果库` 面向已落盘 run snapshot；它不是另一个任务中心，只管理可复查的预测、报告和导入结果。
- `基准集检查` 和 run 检查器共用同一套 `CanvasStage`，画布区域固定在视口内，支持滚轮缩放、拖拽平移、
  label 过滤、对象 hover/active 高亮和样本级预取。
- Benchmark 检查器展示原始 benchmark 的全量 label；Run 检查器展示当前 run 的评估范围，GT、预测、
  label 列表和实例计数都必须按 `target_labels` 过滤。
- 样本检查器侧栏只保留审阅动作：GT/pred、框/线/点和 label 过滤。overlay 样式、label 颜色、
  快捷键等低频配置统一放在 `工作台设置`，避免样本页和设置页双轨维护。
- 样本 payload 同时暴露原图、缩略代理和金字塔瓦片 URL：`image_url` 保留原始文件，viewer 默认使用
  `image_preview_url` 的 JPEG 缩略代理；`image_tile_url_template` 和 `image_tile_size` 作为高倍缩放瓦片边界。
  用户高倍缩放并停顿后，viewer 会延迟加载少量金字塔瓦片增强局部细节。派生图写入
  `eval_bench_store/cache/image_proxy/`，不修改 benchmark/run 原始图片。
- 切换样本时前端会保留上一份 sample detail，并预取当前样本附近的详情；后端缓存 benchmark/run split
  展开的 JSON 路径，避免每次 detail 请求重复读取 split manifest。
- `工作台设置` 的叠图预览优先从当前 benchmark 中选择真实样本，而不是纯手写示意图。所有检查器读取同一份
  overlay、label 颜色和鼠标交互设置。
- `对比分析` 必须把 target labels、per-label delta 和改善/退化样本入口放在报告主体里；样本级对比使用
  `/compare/<baseline>/<candidate>/<sample>` 打开并排 GT/预测检查器。

## 范围

第一版只覆盖当前 Shaft 的视觉结构任务：

- `detection`：预测带 `label` 和 `bbox` 的实例。
- `keypoint`：预测带 `label`、`bbox` 和有序 `keypoints` 的实例。

暂不覆盖通用 VQA、caption 或开放式对话评测。

## 设计原则

- 推理层和 bench 分析层分离。
- Benchmark 由 Eval Bench 管理，不走训练数据 catalog。
- 推理输出必须归一化为接近 raw data 格式的 prediction JSON。
- 已完成的 benchmark manifest、run manifest 和 prediction snapshot 视为不可变产物。
- Metric、parser、prompt 和 visualization 标准通过 spec 版本化。
- Prompt 和推理参数是一等 run snapshot：run manifest 必须保存 prompt ID/path/hash、解析后的 prompt 文本、采样参数、pixel budget，以及 CUDA 设备、tensor parallel size、port、max model length、GPU memory utilization、max sequences 等模型服务参数。
- Eval Bench 负责 benchmark copy、run record、artifact、report、visualization、leaderboard 和 comparison。
- 在相关接口没有被明确迁移前，Shaft 仍然是模型推理 adapter、codec、metric、prompt config 和可视化 helper 的能力真源。

## 企业级任务架构

Eval Bench 不再只按 Control / Execution / Artifact 三层理解；这三层太粗，会让 prompt、metric、job
lifecycle 和 viewer 语义继续漂移。当前正式架构拆成七层，详细约束见
`docs/eval_bench_architecture.md`：

- Presentation Layer：React 页面、workspace layout、dialog、table、viewer panel，只做展示和交互编排。
- API Facade Layer：FastAPI route、request/response 转换、错误响应和日志。
- Control and Lifecycle Layer：SQLite job/service registry、scheduler、resource lease、cancel request 和状态机。
- Execution Layer：worker、runtime adapter、OpenAI/vLLM client、process group cleanup 和 prediction snapshot 落盘。
- Evaluation Semantics Layer：prompt template、target label policy、metric profile、parser/profile 选择。
- Artifact and Store Layer：benchmark copy、run manifest、prediction snapshot、report、comparison 和 trash。
- Rendering and Asset Layer：image proxy、preview/tile、viewer geometry、overlay color/style。

新增能力必须先落在对应中间层：`eval_semantics.py` 统一解析评估语义，`metric_profiles.py`
维护指标 profile，`metrics/` 实现 profile-driven matcher、样本诊断和聚合，`label_policy.py`
维护 target label scope 及来源，`job_lifecycle.py` 维护 job 状态和调度资源占用规则。UI、worker、
evaluator 和 import 不能再各自用 prompt id 字符串推断 layout / arrow / keypoint。

核心对象：

- `Benchmark`：不可变 GT copy，是所有 eval run 的数据真源。
- `Run`：一次模型推理或外部 prediction import 的不可变快照。
- `Job`：可排队的短生命周期工作单元。第一版正式支持 `eval_job`。
- `Runtime`：某个 job 执行时需要的模型后端。`runtime.mode=ephemeral` 表示 job 自己启动并在结束后关闭 vLLM；`runtime.mode=existing_service` 表示连接一个已经存在的 endpoint。
- `Service`：长期运行的模型服务，适合人工调试、共享 endpoint 或多次复用。长期 vLLM 只放在 Services
  页管理，不混入单次 eval job 的生命周期；Jobs 页只管理队列、preflight 和 job record。

Job 采用 manifest-first 设计。前端的 `New job` 只提供模板作为初始值，用户可以在 JSON 中自由添加、修改或删除 runtime / eval 字段。后端提交前会执行 preflight：

- 检查 job kind、benchmark、model path、task、prompt 等必填项。
- 展示将要执行的 vLLM 命令。
- 读取模型 `config.json`，检查 vLLM `tensor-parallel-size` 是否能整除模型 attention heads，并检查
  `CUDA_VISIBLE_DEVICES` 数量是否足够。
- 保留未知 `runtime.args`，并转换成 vLLM CLI flags，例如 `{"limit-mm-per-prompt": {"image": 1}}` 会进入 `--limit-mm-per-prompt '{"image": 1}'`。
- 对未实现的 job kind 不创建队列记录，避免出现永远无法处理的任务。

Prompt 不再只靠用户手写 `prompt_id`。Dashboard 启动时会把 `configs/prompts/grounding_layout.yaml`、`grounding_arrow.yaml`、`keypoint_arrow.yaml` 种子化到 SQLite prompt template registry，默认别名分别是：

- `grounding_layout.latest`
- `grounding_arrow.latest`
- `keypoint_arrow.latest`

Jobs 页可以选择 job template 和 prompt template。默认 `eval_job` 是箭头检测任务；layout 检测保留为
`layout_eval_job` 模板，关键点评估保留为 `keypoint_eval_job` 模板。Prompt template 应用时会把
system prompt、user prompt、parser、metric profile、generation、data 和 `target_labels` 写回
manifest；如果 prompt 没声明目标 label，会清空旧 manifest 上残留的 `target_labels`，避免从 layout
切到 arrow 后仍沿用旧 label 集合。用户也可以直接编辑 manifest 中的 prompt 字段，并把当前 manifest
的 prompt 保存为新的模板。后端 preflight 会用同一份 prompt registry 解析模板；job 入队时保存
resolved manifest，worker 后续执行不依赖前端临时状态。

Detection job 在新建任务面板里提供 label 子任务选择：面板从当前 benchmark summary 的 `labels`、
prompt template 的 `metadata.target_labels` 和 manifest 现有 `target_labels` 合并出候选 chip；
点击 chip 会同步更新 manifest 中的 `eval.target_labels`。`全部候选` 会把当前候选显式写入
`target_labels`；`默认策略` 会删除该字段，让后端继续按 prompt/task 的统一 label policy 解析默认范围。
Keypoint job 不暴露 label 子任务选择，默认只评价 `arrow` 关键点；agent 查询 `resolve-target-labels`
时会返回 `label_subtasks_supported=false`，避免把 keypoint 误当成可任意 label 子集的 detection 子任务。
后端也会在 preflight、init-run、prediction import、worker 和 evaluator 入口拒绝 keypoint 上非 `arrow`
的显式 `target_labels`，防止绕过 UI 直接创建非法子任务；入口级回归测试覆盖 evaluator 与 prediction
import，避免只在前端或单个语义 helper 里保留约束。

每个 prompt template 同时声明 `target_labels`，这是评测语义的一部分，不是纯展示字段。多任务
benchmark 里可能同时有 layout 和 arrow 标注，但一次 run 只应该评价当前 prompt 要求模型输出的
label 集合：

- `grounding_layout.latest` 只评价 `icon / image / shape`。
- `grounding_arrow.latest` 只评价 `arrow`。
- `keypoint_arrow.latest` 只评价 `arrow`，并额外评价关键点。

Evaluator 会在匹配前按 `target_labels` 同时过滤 GT 和 prediction，report summary 和 comparison
也会记录这组 label 以及 `target_labels_source`。`target_labels` 的优先级是：run spec 显式声明、
prompt metadata、legacy prompt ID 兼容推断、task default、unscoped；这套来源会写入 run spec
metadata 并在 report 中保留，避免 prompt/template 推导出的 label 范围被误记为人工显式声明。
这样 layout prompt 在包含 arrow GT 的 benchmark 上评测时，不会把未要求输出的 arrow 误算成漏检；
如果要做全任务评测，应显式使用包含所有目标 label 的 prompt/template，而不是只改
`task=detection`。外部 prediction snapshot 导入时也必须保留这组目标标签，不能只靠
`task=detection` 表达 layout 或 arrow 子任务。

`eval_job` 的最小 manifest 形态：

```json
{
  "kind": "eval_job",
  "runtime": {
    "mode": "ephemeral",
    "engine": "vllm_openai",
    "env": {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    "args": {
      "model": "outputs/qwen3vl-sft/4b/run/best",
      "served-model-name": "qwen3vl-latest",
      "host": "127.0.0.1",
      "port": 8000,
      "tensor-parallel-size": 1,
      "max-model-len": 32768,
      "gpu-memory-utilization": 0.9,
      "max-num-seqs": 8,
      "trust-remote-code": true
    }
  },
  "eval": {
    "model_id": "qwen3vl-latest",
    "benchmark_id": "multitask_val_v1",
    "task": "detection",
    "prompt_id": "grounding_arrow.latest",
    "target_labels": ["arrow"],
    "parser": "raw_data_detection_v1",
    "metric_profile": "detection_iou_v1",
    "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
    "data": {"max_pixels": 1048576, "batch_size": 1}
  }
}
```

Execution Plane 会把 resolved manifest 写入 job metadata 和 run manifest。一次 ephemeral run 的 runtime log 位于 `runs/<run_id>/logs/runtime.log`，推理结束或失败后 worker 会清理自己启动的 vLLM 进程。长期 service 不由 job 自动关闭。

Dashboard API 对所有 4xx/5xx 请求都会写入 `eval_bench_store/logs/backend.log`，并在响应头返回
`X-Eval-Bench-Request-Id`。前端收到失败请求时会弹出 toast，提示后端返回的错误和 request id。
如果 worker 执行 job 失败，job record 会保存错误信息；若已创建 runtime log，也会在 metadata
中保留 `runtime_log_path`，方便从任务队列表格直接定位。

Dashboard 启动后会同时启动一个轻量顶层 orchestrator，不需要用户手动点击“处理下一条”。orchestrator
按固定间隔扫描 queued eval job，结合当前 live job 数、`cuda_visible_devices`、ephemeral runtime
端口和 `tensor_parallel_size` 判断资源是否足够；资源不冲突的任务会被自动 claim 并在后台 worker 线程中
并发执行。默认并发上限为 2，可以通过 `EVAL_BENCH_SCHEDULER_MAX_CONCURRENT_JOBS` 调整；扫描间隔
可通过 `EVAL_BENCH_SCHEDULER_INTERVAL_S` 调整。

Worker 会持续把 `progress_phase`、`progress_done`、`progress_total`、
`progress_current_sample` 和 `progress_message` 写入 job metadata；前端任务中心每 2 秒轮询 job
record 和 scheduler 状态。runtime log 不在主队列表格中常驻展示，点击某条 job row 后才会打开下方
嵌套详情面板，并读取该 job 的完整日志。CLI `process-next-job` 仍保持同步执行，适合在终端或
脚本中使用。

orchestrator claim 新 job 前会检查已有 running job 中记录的 `dashboard_worker_pid`、
`scheduler_pid` 和 `runtime_pid` 是否仍存活。这样即使旧版本 dashboard 进程仍在同步执行长任务，
换新端口打开新版页面也不会误启动第二个冲突评测 job；只有 running 状态已经没有存活进程，或 queued
job 声明的 CUDA 设备和 ephemeral runtime 端口与当前 live job 不冲突时，才允许继续调度。

## Store 布局

默认持久化 store 根目录：

```text
eval_bench_store/
  db/
    eval_bench.sqlite
  logs/
    backend.log
  services/
    <service_id>/
      service.log
  benchmarks/
    <benchmark_id>/
      benchmark.json
      data/
        part1/images/<sample>.png
        part1/json/<sample>.json
      splits/
        val.txt
      previews/
  runs/
    <run_id>/
      run.json
      note.json
      predictions/
        part1/json/<sample>.json
        part2/json/<sample>.json
      raw_outputs/
        <sample>.txt
      previews/
      reports/
      logs/
  tmp/
  exports/
```

`eval_bench_store/` 有意放在 `outputs/` 外面。`outputs/` 继续只承载训练和 checkpoint 产物；Eval Bench 的长期 DB 状态、benchmark copy、run record、preview、report 和 comparison artifact 都放在自己的 store 中。仓库只跟踪 `eval_bench_store/README.md` 和 `eval_bench_store/.gitignore`。

`benchmark.json` 是 benchmark manifest。`run.json` 是 run manifest。`note.json` 是可编辑 run 备注，用来记录复现线索、idea 来源、异常判断和后续排查，不写回不可变 run manifest。每个 prediction JSON 都是带 sample-level metadata 的 raw-data-like prediction document。

## 预测文档格式

```json
{
  "image": "part1/images/example.png",
  "status": "predicted",
  "instances": [
    {
      "label": "arrow",
      "bbox": [100, 120, 420, 180],
      "keypoints": [[110, 150], [420, 150]],
      "extra": {}
    }
  ],
  "metadata": {
    "producer": "eval_bench",
    "run_id": "20260509_000000_demo",
    "model_id": "qwen3vl-sft-best",
    "task": "keypoint",
    "created_at": "2026-05-09T00:00:00Z",
    "latency_ms": 1234.5,
    "inference_params": {},
    "parser": {}
  }
}
```

检测任务的 prediction 可以省略 `keypoints`。关键点 metric 会把缺失 keypoints 视为样本级关键点失败，而不是无效文档结构。

## 当前能力

当前包提供：

- model / benchmark / spec / run / prediction record 的 dataclass schema
- artifact 路径 helper
- 从 `raw_data` validation split 复制 managed benchmark
- JSON 读写与校验工具
- 基于 SQLite 的持久化 job record
- 基于 SQLite 的持久化 prompt template record，支持默认 prompt 种子、Dashboard 选择和从 manifest 保存自定义模板
- 用于创建 benchmark、初始化 run manifest、枚举 benchmark/run/comparison、校验 prediction document、创建/查看 job、执行 job preflight、启动 dashboard、读取 backend log、查看 dashboard state/scheduler snapshot、归档/删除 run、取消/删除 job 和读取 job runtime log 的轻量 CLI/API
- 用于登记、查看、启动、停止、健康探测、删除和日志 tail 的 service registry；也支持只登记外部 vLLM endpoint
- 每个 run 的可编辑 note，Dashboard、API 和 CLI 都可以读写，便于人类与 agent 共享复现上下文
- 独立 Rank Board：后端 `/api/rank-board` 和 CLI `rank-board` 提供同一份排序结果，支持 task、benchmark、status、label、model、prompt、metric、min score、全文 query、facet 计数和多种排序
- Comparison 历史列表：后端 `/api/comparisons`、CLI `list-comparisons` 和前端历史面板共享 task、label、query 过滤语义
- 将模型文本归一化为 prediction document 的 parser 工具，并在写入 snapshot 前按 same-label high-IoU 去重
- 从持久化 prediction snapshot 生成报告的 `evaluate-run`
- React dashboard 外壳：中文工程控制台、总控工作台、benchmark 创建、benchmark sample 浏览、run、持久化 queue 状态、manifest-driven job 创建、job preflight、外部 prediction snapshot 导入、run 评估、run prompt/推理配置查看、样本级 GT/prediction 检查、独立排行榜，以及对比分析；总览只显示粗粒度运营信号和实时状态图，Runs / Rank Board 使用统一高级检索条，主要检查和对比工作区支持可过滤检索
- 两个 run 的对比报告：持久化整体 metric delta 和样本级改善/退化结果，供 dashboard 检查

## 预测结果如何和 test/GT 对比

Eval Bench 不直接拿一次临时输出去扫训练目录。正确流程是：

1. 把 test/val split 复制成一个不可变 benchmark copy。
2. 用模型在这个 benchmark 上推理，输出 raw-data-like prediction snapshot。
3. 对 run 执行 `evaluate-run`，由 evaluator 读取同一份 benchmark GT 和 prediction snapshot，计算 TP/FP/FN、IoU、per-label 指标和样本级诊断。
4. 在 Dashboard 的 Run Inspector 中逐图查看 GT / Prediction 叠图；在 Compare 页比较两个 run 的整体 delta 和 top 改善/退化样本。

也就是说，“test”在 Eval Bench 里对应 `benchmark`，预测结果对应 `runs/<run_id>/predictions/`。二者通过 run manifest 里的 benchmark 引用绑定，不通过文件名猜测或直接读取训练 raw data。

最小命令链：

```bash
.venv/bin/python scripts/eval_bench.py create-benchmark \
  --benchmark-id multitask_test_v1 \
  --task detection \
  --task keypoint \
  --source-root data/raw_data \
  --source-manifest data/raw_data/splits/layout_val.txt \
  --split test \
  --layer layout \
  --layer arrow

.venv/bin/python scripts/eval_bench.py process-next-job
.venv/bin/python scripts/eval_bench.py evaluate-run --run-id <run_id>
.venv/bin/python scripts/eval_bench.py resolve-target-labels \
  --benchmark-id multitask_test_v1 \
  --prompt-id grounding_arrow.latest
.venv/bin/python scripts/eval_bench.py list-runs --task detection --label arrow --limit 20
.venv/bin/python scripts/eval_bench.py rank-board --task detection --label arrow --limit 50
.venv/bin/python scripts/eval_bench.py set-run-note --run-id <run_id> --note-file notes/run.md
.venv/bin/python scripts/eval_bench.py serve-dashboard --host 127.0.0.1 --port 8765
```

如果 run 已经有 prediction JSON，只要它们位于 `eval_bench_store/runs/<run_id>/predictions/`，且路径能通过 image path 映射到 benchmark sample，直接运行 `evaluate-run` 即可重新和 test/GT 对比，不需要重新推理。

如果预测结果还在外部目录，可以直接导入为一个 run 并立即评估：

```bash
.venv/bin/python scripts/eval_bench.py import-predictions \
  --run-id imported_test_predictions \
  --benchmark-id multitask_test_v1 \
  --prediction-root /path/to/prediction_json_dir \
  --task detection \
  --model-id external-model \
  --model-path /path/to/model-or-checkpoint \
  --prompt-id grounding_layout.latest \
  --target-label icon \
  --target-label image \
  --target-label shape
```

`import-predictions` 会按 benchmark split 对齐预测文件，优先匹配同相对路径
（如 `part1/json/a.json`），其次匹配 image path 对应 JSON，最后匹配 basename。
默认允许缺失预测，并在 report 中记录 missing prediction；如果希望缺失时直接失败，
加 `--strict`。如果同一个 `run-id` 已存在，加 `--overwrite` 才会替换。layout 检测导入应使用
`grounding_layout.latest` 或显式传 `--target-label icon --target-label image --target-label shape`；
否则 evaluator 无法知道本轮 detection 是否应排除 arrow。

`init-run` 也支持 `--target-label`，用于直接初始化 detection 的 label 子任务；如果不传，
后端会按同一份 `label_policy.py` 从 prompt id / task 推导默认范围：

```bash
.venv/bin/python scripts/eval_bench.py init-run \
  --run-id layout_icon_eval \
  --task detection \
  --model-id qwen3vl-best \
  --model-path outputs/qwen3vl-sft/run/best \
  --benchmark-id multitask_test_v1 \
  --benchmark-root eval_bench_store/benchmarks/multitask_test_v1/data \
  --benchmark-manifest eval_bench_store/benchmarks/multitask_test_v1/splits/test.txt \
  --split test \
  --spec-id layout.icon \
  --prompt-id grounding_layout.latest \
  --target-label icon
```

Agent 在创建 job 或导入 prediction 之前，可以先用 `resolve-target-labels` 查询同一份 label policy：

```bash
.venv/bin/python scripts/eval_bench.py resolve-target-labels \
  --benchmark-id multitask_test_v1 \
  --prompt-id grounding_arrow.latest
.venv/bin/python scripts/eval_bench.py resolve-target-labels \
  --benchmark-id multitask_test_v1 \
  --task detection \
  --target-label arrow
```

返回会包含 `target_labels`、`target_labels_source`、`candidate_labels`、benchmark/prompt/explicit
三类 label 来源，以及拼错 label 时的 `valid=false` 和 errors；agent 不需要直接读取 benchmark
artifact、prompt registry 或前端 manifest 状态。

Run note 的 agent 入口是稳定 CLI，不需要直接改 store 文件：

```bash
.venv/bin/python scripts/eval_bench.py get-run-note --run-id imported_test_predictions
.venv/bin/python scripts/eval_bench.py set-run-note \
  --run-id imported_test_predictions \
  --note "reproduce: ckpt epoch_3; idea: prompt v2"
```

Agent 的生命周期操作也走稳定 CLI，不需要手改 SQLite 或移动 artifact 目录：

```bash
.venv/bin/python scripts/eval_bench.py dashboard-state
.venv/bin/python scripts/eval_bench.py scheduler-status
.venv/bin/python scripts/eval_bench.py backend-logs --max-lines 200
.venv/bin/python scripts/eval_bench.py archive-run --run-id imported_test_predictions
.venv/bin/python scripts/eval_bench.py delete-run --run-id imported_test_predictions
.venv/bin/python scripts/eval_bench.py cancel-job --job-id <job_id>
.venv/bin/python scripts/eval_bench.py job-logs --job-id <job_id> --max-lines 200
.venv/bin/python scripts/eval_bench.py delete-job --job-id <job_id>
.venv/bin/python scripts/eval_bench.py delete-service --service-id local-vllm-0
```

`archive-run` / `delete-run` 复用 `EvalBenchStore`；`cancel-job` / `delete-job` 复用
`EvalBenchDatabase`；`job-logs` / `backend-logs` 复用后端 `log_utils.py`。删除 run 和 service
默认会通过 `StoreLayout.move_to_trash` 保留证据目录。

Agent 检索基础对象走稳定 CLI/API，不需要读取前端 state 或手扫 store 目录。CLI 的
`list-benchmarks` / `show-benchmark` / `list-runs` 与 API 的 `GET /api/benchmarks` /
`GET /api/runs` 共享 task、label、model、prompt、metric 和全文查询语义；job、service 和
comparison 列表同样由后端分页过滤，单个 job / service 详情用 `show-job` / `show-service`
读取。Job template 和 prompt template 也有 CLI 入口，agent 创建 job 前可以先发现模板、读取单个模板、
筛选任务类型，并按需维护 prompt template registry。CLI 模块本身保持轻量 import；dashboard、worker、
evaluator 和模型运行时依赖只在具体命令执行时懒加载，避免 agent 的检索入口被重型运行时拖慢：
`list-agent-commands` 会输出当前稳定 agent 命令面；真实分发集中在 `eval_bench.cli._command_handlers()`。
每条命令还会带 `domain`、`mutates_state`、`arguments` 和 `mutually_exclusive_groups`，便于 agent 区分
只读查询、artifact 写入和 service/job 生命周期操作，并直接读取参数名、flag、类型、默认值、choices、
是否 repeatable 和互斥组要求；同时包含顶层 `recommended_runner`、每条命令的稳定 `argv_prefix` 和单行
`usage`，agent 可以直接组合 argv，不需要从自然语言 help 里猜命令形态。`mutates_state`
只是副作用标记，不是权限控制。新增 agent 命令必须同时进入
parser、handler 映射和 `AGENT_COMMAND_METADATA`，`AGENT_STABLE_COMMANDS` 由 metadata 派生，测试会检查这些集合
一致，避免 agent 看到 help 里的命令却无法通过真实入口执行、无法判断副作用或仍要从自然语言 help 猜参数。

```bash
.venv/bin/python scripts/eval_bench.py list-agent-commands
.venv/bin/python scripts/eval_bench.py list-job-templates --query keypoint
.venv/bin/python scripts/eval_bench.py show-job-template \
  --template-id keypoint_eval_job
.venv/bin/python scripts/eval_bench.py list-prompt-templates \
  --task detection \
  --query arrow
.venv/bin/python scripts/eval_bench.py show-prompt-template \
  --prompt-id grounding_arrow.latest
.venv/bin/python scripts/eval_bench.py upsert-prompt-template \
  --payload-file /path/to/prompt_template.json
.venv/bin/python scripts/eval_bench.py delete-prompt-template \
  --prompt-id custom.arrow.v1
.venv/bin/python scripts/eval_bench.py list-benchmarks \
  --task detection \
  --layer layout \
  --split val \
  --query multitask
.venv/bin/python scripts/eval_bench.py show-benchmark \
  --benchmark-id multitask_test_v1
.venv/bin/python scripts/eval_bench.py list-runs \
  --task detection \
  --benchmark-id multitask_test_v1 \
  --label arrow \
  --model-id qwen3vl-best \
  --metric-profile detection_iou_v1 \
  --query "prompt v2"
.venv/bin/python scripts/eval_bench.py list-comparisons \
  --task detection \
  --label arrow \
  --query qwen3vl
.venv/bin/python scripts/eval_bench.py show-comparison \
  --baseline-run-id old_run \
  --candidate-run-id new_run
.venv/bin/python scripts/eval_bench.py show-comparison-sample \
  --baseline-run-id old_run \
  --candidate-run-id new_run \
  --sample-index 0
.venv/bin/python scripts/eval_bench.py list-jobs \
  --kind eval \
  --status queued \
  --query qwen3vl
.venv/bin/python scripts/eval_bench.py show-job \
  --job-id eval_20260513_103418_ebb7f052
.venv/bin/python scripts/eval_bench.py list-services \
  --kind local_vllm \
  --status running \
  --query qwen3vl
.venv/bin/python scripts/eval_bench.py show-service \
  --service-id local-vllm-0
```

Agent 排障也走稳定 CLI，不需要直接读取 `eval_bench_store`。以下命令分别对应 dashboard
中的 run summary、report、run sample inspector 和 benchmark inspector；run sample 输出会按
当前 run 的 `target_labels` 自动裁剪 GT、prediction、payload 和 diagnostics：

```bash
.venv/bin/python scripts/eval_bench.py show-run --run-id <run_id>
.venv/bin/python scripts/eval_bench.py show-run-report --run-id <run_id> --summary
.venv/bin/python scripts/eval_bench.py list-run-samples \
  --run-id <run_id> \
  --label arrow \
  --error-filter fn
.venv/bin/python scripts/eval_bench.py show-run-sample \
  --run-id <run_id> \
  --sample-index 12
.venv/bin/python scripts/eval_bench.py list-benchmark-samples \
  --benchmark-id multitask_test_v1 \
  --label arrow
.venv/bin/python scripts/eval_bench.py show-benchmark-sample \
  --benchmark-id multitask_test_v1 \
  --sample-index 12
```

Rank Board 已从 Compare 页拆出为独立工作台。Dashboard 入口是左侧“排行榜”，API 入口是
`GET /api/rank-board`，CLI 入口是：

```bash
.venv/bin/python scripts/eval_bench.py rank-board \
  --task detection \
  --benchmark-id multitask_test_v1 \
  --label arrow \
  --metric-profile detection_iou_v1 \
  --min-score 0.70 \
  --sort-by f1_iou50 \
  --sort-order desc \
  --query "prompt v2"
```

显式加权排行只在用户或 agent 传入 `rank_scheme` 时启用；默认 Rank Board 不做加权，仍按
`f1_iou50` 排序。非加权模式下，`--sort-by precision_iou50|recall_iou50|mean_iou|prediction_count`
会同步更新返回的 `primary_metric`、`primary_metric_label` 和 entry `score`；`created_at` / `run_id`
只作为列表排序维度，主指标保持 F1。权重项必须声明 benchmark、metric、weight 和缺失指标处理规则：

```bash
.venv/bin/python scripts/eval_bench.py rank-board \
  --rank-scheme-json '{
    "name": "arrow_quality_v1",
    "terms": [
      {"benchmark_id": "multitask_test_v1", "metric": "f1_iou50", "weight": 0.7, "missing": "drop"},
      {"benchmark_id": "multitask_test_v1", "metric": "mean_iou", "weight": 0.3, "missing": "zero"}
    ]
  }'
```

Dashboard 的 Rank Board 也提供折叠式 `Weighted rank scheme` 面板，可以粘贴同一份 JSON；启用后页面会
显示 weighted score、score components 和后端返回的 score formula。未启用或 JSON 未通过前端基础校验时，
页面继续保持默认 F1 主指标，不把 weighted score 混写成默认分数；如果后端拒绝 scheme，错误会留在
weighted 面板内，排行榜表格继续显示上一份可用结果。

Dashboard 总览页是总控工作台，不展示 recall 这类精细评测指标，也不把状态分布拆成一墙低价值面板。
首页只回答三件事：当前是否可用、评测管线卡在哪里、下一步应该去哪个页面。总览视觉模块按两列 command deck
处理：左侧 focus panel 用下一步动作、四个可行动信号、Benchmarks/Predictions/Evaluated/Rank Ready 管线
和 Run/Job/Service 三泳道 12 日期桶活动矩阵承载核心判断；右侧只保留行动入口和最近 run
的可点击紧凑摘要。行动入口必须以 readiness switchboard 形态展示 service、queue、evaluation 和 rank board
四个状态入口，每个入口给出状态、数量占比和目标页面，不能退化成普通链接列表；最近 run 按 `created_at`
倒序截取，不依赖 API 返回顺序。不再维护分栏 masonry 图表墙，也不要求固定数量的 mini chart。
Notes、任务类型、模型分布、benchmark task、label footprint、样本/label 权重、Job 日历和 scheduler 资源不进入总览，
留在 Runs、Inspector、Rank Board 或 Services 的细节视图里。compact 视口下由 command deck 自身滚动，不能被外层 hidden 裁切。首页和标准 workspace 元素保留轻量入场、hover、状态 pulse 与 rail 动效，这些动效只用于强调状态、可点击入口和实时感，不改变数据语义。高级检索 UI 已组件化为
`AdvancedFilterBar`，默认收敛为一个 Filter 按钮，展开后才显示检索表单；Benchmarks、Jobs、Services、Runs、Compare 和 Rank Board 共享这套筛选布局；
Benchmark Inspector 和 Run Inspector 的样本级 label/error 筛选也复用同一个折叠式高级检索条，避免侧栏堆叠多个 select；
当任意条件生效时，`AdvancedFilterBar` 会显示统一的“清空”动作；search/number 默认清空，带 `all`
选项的 select 回到 `all`，排序这类没有 `all` 的 select 回到第一个默认值，避免默认排序被误算成过滤条件。
排行榜入口、入榜状态和已评估状态使用独立 `AppIcon` 语义 key，避免用同一个 metrics 图标表达不同含义。
Runs 页和 Compare 页的 run 高级检索直接请求 `GET /api/runs`，和 CLI `list-runs` 共享 task、benchmark、status、
label、model、prompt、metric 和 note 全文查询语义；Runs、Compare、Benchmarks、Jobs、Services 和 Rank Board 都必须走后端分页，筛选变化回到第一页，
不能在前端固定读取 200 条后截断；Benchmarks 页直接请求 `GET /api/benchmarks`，
和 CLI `list-benchmarks` 共享 task、layer、split 和全文查询语义；Jobs 页直接请求 `GET /api/jobs`，
和 CLI `list-jobs` 共享 kind、status、query、offset 和 limit 语义；Services 页直接请求 `GET /api/services`，
和 CLI `list-services` 共享 kind、status、query、offset 和 limit 语义；Rank Board 前端和 CLI/API 默认用
`f1_iou50` 作为主指标排序，用户可以把主指标切到 precision、recall、mIoU 或预测数，也可以按创建时间或 run id 排列列表；
Rank Board 页面使用后端 `offset/limit` 分页请求，不再只取固定前 200 条，翻页时保留当前筛选、排序和 weighted scheme；
显式 weighted scheme 可通过 CLI/API 或前端折叠面板传入，会返回 `weighted_score`、`rank_scheme` 和每条
entry 的 `score_components`，便于 agent 解释最终分。
Benchmark summary 会暴露 `labels`，供任务创建、检索 facet 和 agent CLI 统一消费，不要求前端扫描
benchmark 文件。
后续页面新增 filter 应优先复用；页面局部下拉输入优先复用 `controlPrimitives.tsx`，不能在业务页直接复制
`filter-select compact` 或 raw `<select>`。弹窗统一走 `WorkspaceDialog`，关闭按钮、Escape、backdrop 点击和内部滚动都由
该组件管理。页面标准动作统一走 `ActionButton`、`CommandButton` 或 `IconActionButton`，业务页只为
画布 HUD 这类低层交互保留专用样式；viewer 图层预设使用 `CompactSelectControl`，
label chip 使用 `OptionChipButton`；Compare label delta 和 viewer object row 这类可选卡片使用 `SelectableCardButton`；
弹窗表单和对比选择轨的 select 使用 `FormSelectControl`；
弹窗表单提交和 Settings 快捷键捕获控件都直接使用 `ActionButton`，可以保留 `shortcut-capture`
这类专用外观 class，但不保留页面私有 raw button。
`test:ui-contracts` 会阻止原生 confirm/alert/prompt、
业务页自建 dialog shell、旧 `sample-filters` 和已收敛标准动作回流。

Dashboard 的 Runs 页也提供同一能力：展开 `Import prediction snapshot`，填写
`run_id`、benchmark、prediction root、task、model ID 和可选 target labels 后提交。target labels
留空时不由前端补默认值，后端按 prompt/task 的同一套 `label_policy.py` 解析；显式填写时作为
detection 的 label 子任务范围。后端会创建标准 run、复制 prediction snapshot、默认立即评估，
之后直接进入 Run Inspector 看逐图 GT / Prediction 对比。Run Inspector 顶部的记录配置面板可以编辑
同一份 run note；Runs 表格会把 note 摘要纳入搜索。

Dashboard 的 Benchmarks 页也可以创建 benchmark copy：打开 `创建副本` 弹层，
填写 raw_data 根目录、split manifest、任务类型和 layer 后提交。这个操作与
`create-benchmark` CLI 使用同一套后端逻辑，创建后再进入 benchmark inspector 检查 GT。

## 开发

在 Shaft 仓库根目录执行：

```bash
uv pip install -e ".[dev,eval-bench]"
.venv/bin/python scripts/eval_bench.py --help
PYTHONPATH=projects/eval_bench .venv/bin/python -m pytest -q projects/eval_bench/tests
```

前端依赖由 frontend package 自己管理：

```bash
cd projects/eval_bench/frontend
npm install
npm run build
```

在仓库根目录启动已构建 dashboard：

```bash
.venv/bin/python scripts/eval_bench.py serve-dashboard --host 127.0.0.1 --port 8765
```

Dashboard 服务启动后，运行无头渲染检查：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run render-check
```

核心页面、窄屏布局和弹窗滚动边界也有独立 smoke；它会遍历 overview、rank-board、runs、benchmarks、
jobs、services 和 settings，在 desktop / compact / narrow 三种 viewport 下确认没有全局滚动溢出、
page stack 不被 content 裁切、表格和高级检索面板需要滚动时由自己的容器滚动，并确认 rank-board
和 compare 加载独立 chunk。Overview 还会额外检查顶栏 status 是独立圆角 capsule、没有外层容器，
Run/Job/Service 活动矩阵保持 3 条 12 桶泳道，Overview 不能回流旧 mini chart wall 或 chart matrix；页面必须保留 next action、四段管线、readiness switchboard、一个 focus panel 和最近 run 紧凑摘要组成的 command deck，
并保留报告覆盖、待评估、队列压力和在线服务四个可点击信号，不再回退到 Run/Ops/Volume 面板组；
不出现 precision/recall/IoU 这类细指标文案，也不能回流 Notes、Label footprint、模型分布、Job 日历或 Scheduler 资源这类低价值总览面板，
关键入口必须有 hover/transition 反馈。Runs 页必须暴露 `.run-list-pager` 并通过 `/api/runs?offset&limit` 分页，
Compare 页必须暴露 `.compare-run-pager` 并通过 `/api/runs?offset&limit` 分页候选 run，
翻页时不能清空 URL 或上一页选中的 baseline/candidate；Benchmarks 页必须暴露 `.benchmark-list-pager` 并通过 `/api/benchmarks?offset&limit` 分页，
Jobs 页必须暴露 `.job-list-pager` 并通过 `/api/jobs?offset&limit` 分页，
Services 页必须暴露 `.service-list-pager` 并通过 `/api/services?offset&limit` 分页，
Rank Board 必须暴露独立 pager。Benchmark / Run 检查器还会模拟样本过滤后 0 命中的状态，确认侧栏、
高级检索按钮和空结果提示仍在原工作区内可见，不能退化成全页 EmptyState；检查器还会在窄屏 split
堆叠时检查 `.image-stage` 不能塌缩，若侧栏、画布和对象检查器无法同时塞进一屏，必须由
`.visual-inspector-page` 自己滚动，而不是外层 hidden 裁切。Runs / Rank Board 会实际输入筛选再点击清空，确认默认排序不被误算成生效条件：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:layout
```

开启画布交互 smoke 后运行同一检查：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/runs/<run_id>?sample=0 \
  INTERACTION_SMOKE=1 \
  npm run render-check
```

`INTERACTION_SMOKE=1` 会实际操作查询筛选、标签 / 图层开关、滚轮缩放、画布拖拽、对象列表点击和基础快捷键。评测检查器的交互设计以图像工作台为主：桌面优先固定在一个屏幕内，样本列表、对象列表和表格只在自己的面板内滚动；compact / narrow 视口下如果 split 面板纵向堆叠，页面本身可以滚动，但画布必须保留可操作高度，不能被对象检查器或样本列表压成不可见；配置、prompt 和标签指标明细默认折叠，低频的创建/注册表单不抢占主画布。前端 metric 中间层还有独立检查：

```bash
cd projects/eval_bench/frontend
npm run test:metrics
npm run test:status-model
npm run test:workspace-settings
npm run test:ui-contracts
npm run test:shortcuts
npm run test:layout
```

`test:status-model` 会检查 job、run、service 的状态文案、视觉 tone 和可执行动作规则，避免组件自己临时
判断“能不能取消、删除、启动、停止、评估”。
`test:ui-contracts` 会静态检查 UI 组件边界，避免业务页重新引入阻塞式浏览器弹窗、直接写 dialog 外壳或
绕过标准 action button；Settings、Runs、Services 和 Compare 的局部 select、搜索清空、重置和 label
清除动作也在这个边界内。

`test:workspace-settings` 会检查 viewer/settings 共享配置 schema，确保数值配置项、UI number input 范围、
归一化范围和显示缩放系数来自同一份定义，避免配置层级再次分叉。

`test:shortcuts` 会做两层覆盖：静态扫描所有全局 `keydown` 入口必须经由 `SHORTCUT_ACTIONS` /
`useWorkspaceShortcuts`，并用自定义 keymap 在 benchmark、run、compare 和 settings 页面真实触发
样本切换、图层显隐、几何显隐、视图复位、清除选择和快捷键编辑，确认旧默认键不会绕过用户配置。

针对 viewer 高频鼠标交互还有独立性能检查；它会在真实 run inspector 上连续执行滚轮缩放和拖拽平移，
并确认 pan/zoom 不会重渲染 heavy overlay layer：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/runs/<run_id> npm run test:viewer-performance
```

工作台设置页的叠图预览也有单独回归；它会检查默认 role 色、label × role 颜色和线宽是否实时反映到预览，以及 label 颜色匹配是否大小写不敏感：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/settings npm run test:settings-preview
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:shortcuts
```

Dashboard v2 的基本操作约定：

- 评测记录页是 run 管理中心：支持 run、模型、基准集搜索，支持状态、任务、基准集筛选，两条 run 勾选后直接进入对比页，也可以重新评估、归档和删除。
- 任务中心页是队列管理中心：Dashboard 后台 orchestrator 会自动推进 queued eval job；表格展示 scheduler 状态、实时阶段、进度条和当前 sample。runtime log 不常驻占用列表空间，点击某条 job row 后会在嵌套详情面板中查看完整日志。排队中的 job 可以取消，运行中的 job 可以终止；终止会写入 `cancel_requested` 并尝试结束 job 专属 ephemeral runtime 进程组。无效、失败、已取消或 demo job 可以删除。删除记录会写入 `eval_bench_store/trash/`，默认不硬删证据。
- 模型服务页管理模型服务 registry：本地服务可启动/停止，服务记录可删除；删除前会先尝试停止运行中的本地服务。
- 工作台设置页维护浏览器本地用户偏好：label × role 颜色矩阵、框线宽、骨架线宽、点半径、标签字号、描边、透明度、预测线型、滚轮缩放灵敏度、拖拽平移灵敏度、缩放上下限和快捷键映射会写入本地设置，所有检查器自动读取；页面采用顶部 command bar + 大画布预览 + 底部 Preferences 抽屉，预览优先使用真实 benchmark sample，只有缺少可绘制样本时才回退到 `projects/eval_bench/static/settings_preview.svg`。
- 前端业务状态由 `frontend/src/statusModel.ts` 统一维护；job、run、service 的 badge 文案、live 动画和操作按钮启用条件都读取同一份状态模型。
- 当前版本没有权限系统；顶栏和设置页只展示 `local` / `Local Browser` profile，明确这些偏好属于当前浏览器本地用户配置。
- 左侧导航栏支持通过顶部图标按钮收起/展开；折叠状态也写入浏览器本地设置，便于大屏评测时把主画布面积留给工作区。
- 评测检查器快捷键由 `evalBench.shortcuts` action map 管理；Settings 页展示命令 action、当前键位和冲突状态，键位支持 `Ctrl` / `Alt` / `Shift` / `Meta` 组合。新增图层或工具时只需要注册新的 action，不在文档和页面里写死具体类别。
- 新增任何全局键盘交互时，必须先加入 `SHORTCUT_ACTIONS`，再在 `test:shortcuts` 覆盖对应页面或组件；局部输入控件只允许保留自己的局部键，例如 label 添加框的 Enter。

## Dashboard 工作区交互规范

Eval Bench 的 dashboard 按工程工作台设计，不按长页面表单设计。页面主区域应该尽量固定在一个屏幕内，
低频配置折叠到详情面板，用户主要通过画布、列表和表格完成排障。

当前检查器工作区遵循这些约定：

- 基准集检查、run 检查和工作台设置页的叠图预览复用同一个 `CanvasStage`，不要再各自实现独立画布。
- 图片在容器内按原始宽高比自适应显示，默认完整可见；缩放范围由 `workspaceSettings.ts` 统一管理，默认 25% 到 800%，可在工作台设置中调整。
- 鼠标滚轮缩放是连续缩放；左键拖拽平移。滚轮和平移灵敏度由工作台设置统一管理，缩小到 100% 以下时也允许拖拽，便于用户调整画布位置。
- 样本列表 / 主画布、主画布 / 对象检查器、Compare 工作区、成对样本对比和 Job manifest 编辑/预检查区都使用可拖拽分栏，拖拽后的宽度写入浏览器本地设置；Settings 页面不再使用三栏分割，避免预览被配置面板挤压。窄屏下可拖拽分栏会隐藏 resizer 并纵向堆叠；如果内容高度超过可视区，检查器页面负责滚动，画布和对象检查器各自保留稳定高度。
- 目录分页统一使用 `PagerControl` 和 `clampListPageOffset`；页面只保留自己的 page size、查询参数和业务 className，不允许复制上一页/下一页禁用逻辑。
- 样本过滤后即使没有命中，也必须保留检查器侧栏和高级检索入口；只能在样本列表和主画布区域显示空结果，避免用户无法撤销过滤。
- 新建评测、创建 benchmark、导入 prediction snapshot 和登记 service 属于低频操作，只能通过临时弹层打开；
  Jobs / Benchmarks / Runs / Services 主页面不再使用嵌套 tab 放表单。
- 设置里的连续数值项使用紧凑 number input，不再使用横向 slider；这类配置强调精确和可压缩，而不是拖动演示。
- 导航、总览指标和主操作按钮使用 image_gen 生成并裁剪的业务 PNG 图标库；关闭、删除、搜索等工具动作继续使用矢量图标。
- 按钮层级、按钮图标和弹层表单网格规则见 `docs/eval_bench_ui_icon_design.md`；标准动作必须复用
  `ui.tsx` 中的 `ActionButton`、`CommandButton` 或 `IconActionButton`，可选行、chip 和 card 必须优先复用
  `SelectableRowButton`、`OptionChipButton` 和 `SelectableCardButton`，局部 select 必须优先复用
  `controlPrimitives.tsx` 的 `CompactSelectControl` 或 `FormSelectControl`。长路径/URL 字段必须使用
  `wide-field`，结果和错误区使用 `full-field`，不能把所有输入框强行压进等宽列。
- 样本列表会预加载当前样本前后若干张图，避免翻页或键盘切换时出现明显空白等待。
- 样本卡片、对象列表和诊断文本默认允许换行；不能用省略号隐藏 image path、label、IoU 或 bbox 等排障信息。
- 样本跳转输入框已取消。按样本浏览使用列表、分页和快捷键 action map；精确定位后续应通过查询/过滤能力完成，而不是在侧栏常驻一个窄输入框。
- 表格行选择使用显式 checkbox 样式，不依赖浏览器默认外观；选择列不应出现无语义装饰点。

叠图实例颜色按笛卡尔积配置：

- 基础 role：`GT`、`Pred`、`FN`、`FP` 各有固定默认颜色；active highlight 只作为当前高亮描边，不参与实例底色判定。
- Label × role：用户可以为任意 label 的任意 role 配置颜色，例如 `arrow × GT` 和 `arrow × Pred` 是两个独立单元格。没有显式单元格配置时，回退到该 role 的默认颜色。

Arrow / connector 的 linestrip 可视化必须表达方向：骨干线之外还要画起点、终点和位于中间线段上的自适应方向三角形。没有
linestrip 的 arrow 只画 bbox；有 linestrip 的 arrow 同时画 bbox 和骨干，不需要额外的 zoom preview。

从 `raw_data` validation split 创建 benchmark copy：

```bash
.venv/bin/python scripts/eval_bench.py create-benchmark \
  --benchmark-id multitask_val_v1 \
  --task detection \
  --task keypoint \
  --source-root data/raw_data \
  --source-manifest data/raw_data/splits/layout_val.txt \
  --split val \
  --layer layout \
  --layer arrow
```

创建并查看持久化 eval job。Dashboard 推荐路径是在任务中心页打开“新建评测”弹层，选择模板后直接编辑
manifest，先点“预检查”查看 preflight，再提交。CLI 使用同一套 prompt registry 和 preflight 逻辑，
agent 可以先校验再入队，不需要手写 SQLite 或绕过 API：

```bash
.venv/bin/python scripts/eval_bench.py preflight-job \
  --payload-json '{"manifest":{"kind":"eval_job","runtime":{"mode":"ephemeral","engine":"vllm_openai","env":{"CUDA_VISIBLE_DEVICES":"0","CUDA_DEVICE_ORDER":"PCI_BUS_ID"},"args":{"model":"outputs/qwen3vl-sft/run/best","served-model-name":"qwen3vl-best","host":"127.0.0.1","port":8000,"tensor-parallel-size":1,"max-model-len":32768,"gpu-memory-utilization":0.9,"max-num-seqs":8,"trust-remote-code":true}},"eval":{"model_id":"qwen3vl-best","benchmark_id":"multitask_val_v1","task":"detection","prompt_id":"grounding_arrow.latest","target_labels":["arrow"],"generation":{"max_tokens":4096,"temperature":0,"top_p":1},"data":{"max_pixels":1048576,"batch_size":1}}}}'

.venv/bin/python scripts/eval_bench.py create-job \
  --payload-json '{"manifest":{"kind":"eval_job","runtime":{"mode":"ephemeral","engine":"vllm_openai","env":{"CUDA_VISIBLE_DEVICES":"0","CUDA_DEVICE_ORDER":"PCI_BUS_ID"},"args":{"model":"outputs/qwen3vl-sft/run/best","served-model-name":"qwen3vl-best","host":"127.0.0.1","port":8000,"tensor-parallel-size":1,"max-model-len":32768,"gpu-memory-utilization":0.9,"max-num-seqs":8,"trust-remote-code":true}},"eval":{"model_id":"qwen3vl-best","benchmark_id":"multitask_val_v1","task":"detection","prompt_id":"grounding_arrow.latest","target_labels":["arrow"],"prompt_path":"configs/prompts/grounding_arrow.yaml","generation":{"max_tokens":4096,"temperature":0,"top_p":1},"data":{"max_pixels":1048576,"batch_size":1}}}}'

.venv/bin/python scripts/eval_bench.py list-jobs
```

登记本地 vLLM OpenAI server：

```bash
.venv/bin/python scripts/eval_bench.py register-service \
  --kind local_vllm \
  --service-id local-vllm-0 \
  --model-path outputs/qwen3vl-sft/run/best \
  --served-model-name qwen3vl-best \
  --cuda-visible-devices 0 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16

.venv/bin/python scripts/eval_bench.py list-services \
  --kind local_vllm \
  --status registered \
  --query qwen3vl
.venv/bin/python scripts/eval_bench.py service-command --service-id local-vllm-0
```

本地服务可以由 Eval Bench 启停。启动时使用当前 `.venv` 的 Python 执行 `python -m vllm.entrypoints.openai.api_server`，日志写入 `eval_bench_store/services/<service_id>/service.log`。外部服务只登记 endpoint，不由 Eval Bench 启停。

```bash
.venv/bin/python scripts/eval_bench.py start-service --service-id local-vllm-0
.venv/bin/python scripts/eval_bench.py service-health --service-id local-vllm-0
.venv/bin/python scripts/eval_bench.py service-logs --service-id local-vllm-0 --max-lines 200
.venv/bin/python scripts/eval_bench.py stop-service --service-id local-vllm-0
.venv/bin/python scripts/eval_bench.py delete-service --service-id local-vllm-0
```

`start-service` 只负责拉起本地进程，状态会先进入 `starting`；`service-health` 会探测 OpenAI-compatible `/v1/models` endpoint，探测成功后状态变为 `running`，探测失败但本地进程还活时保持 `starting`，进程不存在则变为 `stopped`。Dashboard 的 Services 页复用同一套接口，显示 health、最近检查时间、错误信息和日志 tail。

处理下一个 queued eval job。CLI 会同步执行完整 worker；Dashboard API 则后台启动 worker 并立即返回
running job，供前端轮询状态和进度：

```bash
.venv/bin/python scripts/eval_bench.py process-next-job
.venv/bin/python scripts/eval_bench.py cancel-job --job-id <job_id>
.venv/bin/python scripts/eval_bench.py job-logs --job-id <job_id> --max-lines 200
```

第一个 worker action 有意命名为 `prepare_run`：它会 claim 一个 queued job，写入 `eval_bench_store/runs/<run_id>/run.json`，把 run manifest path 记录到 job metadata 中，并把真实模型推理留给后续 worker layer。

当 job manifest 使用 `runtime.mode=ephemeral` 且 `runtime.engine=vllm_openai` 时，worker 会先启动一个该 job 专属的 vLLM OpenAI server，等待 `/v1/models` ready，再执行推理，最后关闭这个进程。runtime stdout/stderr 写入 `runs/<run_id>/logs/runtime.log`，Dashboard 可通过 `/api/jobs/<job_id>/logs` 读取 tail。当 job manifest 使用 `runtime.mode=existing_service` 并提供 `endpoint` 时，worker 不负责启停模型服务，只通过 OpenAI-compatible chat-completions 请求执行推理。`endpoint` 可以是 server root、`/v1` 或 `/v1/chat/completions`。

做集成测试时，可以在 job payload 中设置 `"backend":"dry_run"`。worker 会为每个 benchmark sample 写入空 prediction snapshot，执行 `evaluate-run`，并把 run 标记为 `succeeded`。这可以在不调用模型后端的情况下验证完整 artifact 链路。

评估已有 run 的 prediction snapshot：

```bash
.venv/bin/python scripts/eval_bench.py evaluate-run --run-id <run_id>
```

第一份 metric report 会写入 `runs/<run_id>/reports/metrics.json`，内容包括 IoU@0.50 precision/recall、mean IoU、per-label count、缺失 prediction file，以及 keypoint 任务下的 endpoint distance。检测任务使用 `detection_iou_v1` 的 per-label bbox IoU matcher；关键点任务使用 `keypoint_endpoint_v1` 的有序起终点距离 matcher，bbox IoU 只保留为诊断字段，不再决定 TP/FP/FN。它也会保存样本级诊断信息：TP/FP/FN 数量、match pair、per-sample mean IoU、false positive reference 和 false negative reference。`evaluate-run` 还会写入 `runs/<run_id>/reports/summary.json`，这是不含 sample list 的小型 dashboard index 文件，避免频繁刷新的 run table 解析完整 diagnostic report。

Dashboard 的基准集检查器会直接读取 copied benchmark GT，并在 copied image 上叠加实例，支持服务端分页和标签过滤。评测检查器会读取同一份 benchmark GT 与 run prediction snapshot，叠加真值 / 预测的 bbox、linestrip 和 keypoint，并展示样本级诊断信息；它支持服务端分页、错误类型过滤和标签过滤，因此视觉排障不需要重新跑推理，也不需要手工打开生成文件。

Sample viewer 的目标更接近标注工具，而不是静态 preview：用户可以过滤 label，切换真值 / 预测以及框 / 线 / 点图层，调节任意 label × role 颜色规则、框线宽、骨架线宽、点半径、标签字号、高亮宽度、透明度、预测线型、鼠标交互参数和快捷键映射，hover 或 click 某个 object 时会在对象列表和叠图中同步高亮，也可以使用滚轮缩放和拖拽平移查看局部。label 和图层显示偏好会写入浏览器本地状态，run inspector 翻页或刷新样本详情时只清理 hover/lock 这类样本临时交互。label 颜色由用户手动添加，运行时按实际 label 和 role 匹配，不预设 `arrow/icon/text` 这类固定任务名；linestrip 会绘制起点、终点和位于中间线段上的自适应方向三角形。Run inspection、工作台设置预览和成对样本对比复用同一个 viewer。工作台设置预览只替换底图和 demo instances，不另写独立可视化逻辑。原生 wheel listener 由 React effect 清理，sample list 走服务端分页并预加载当前样本附近的图片，以保证大 benchmark 下 dashboard 仍然流畅。

Metric 展示分三层：

- 总览：当前可见 label 下只显示真实实例数和预测实例数，保持检查器主视图的低噪声密度。
- 分 label 明细：每个 label 的 GT、Pred、TP、FP、FN、P@0.50、R@0.50 和 mean IoU。
- 对象级诊断：每个 GT/Pred object 都显示 TP/FP/FN/unchecked 状态、匹配的对侧 object index、IoU 和 bbox 坐标。组件不直接推导这些状态，统一由 `viewerMetrics.ts` 计算，便于后续扩展新的任务类型、匹配策略和 object schema。

比较两个已经评估过的 run：

```bash
.venv/bin/python scripts/eval_bench.py compare-runs \
  --baseline-run-id <old_run_id> \
  --candidate-run-id <new_run_id>
```

Comparison report 会写入 `eval_bench_store/exports/comparisons/`。它只比较已经持久化的 metric report，不重新跑推理。Dashboard 的 Compare 页通过 `/api/comparisons` 读取同一份 report，展示 P/R/IoU、endpoint distance、TP/FP/FN 和 endpoint pair delta；样本和标签排行使用 metric profile 保留的主指标语义，因此 `keypoint_endpoint_v1` 中 endpoint distance 下降会被视为改善。Compare 页同时列出已保存 comparison，并提供 top 改善/退化样本到并排样本对比 viewer 的跳转。Rank Board 负责全局排名、facet 和主指标排序，默认主指标是 F1@.50。Compare 工作区和成对样本对比的左右 run 面板都使用可拖拽分栏，适合在不同屏幕宽度下长期排障。
Agent 读取已保存 comparison 用 `show-comparison`；读取成对样本详情用 `show-comparison-sample`，两者都走
store/comparison API，不需要直接读取 `exports/comparisons` 或 run artifact 文件。

运行轻量 dashboard-store performance smoke：

```bash
.venv/bin/python scripts/eval_bench.py perf-smoke --iterations 5 --sample-limit 500
```

这个命令会测量 dashboard 常用路径：state summary、saved comparison listing、benchmark sample listing，以及存在 run 时的 run sample listing。它是本地迭代的 smoke signal，不替代更重的 load testing。

`scripts/eval_bench.py` 是薄包装入口，只负责把 `projects/eval_bench` 加入 `sys.path`，然后调用 `eval_bench.cli`。共享依赖继续维护在根目录 `eval-bench` extra 中，不要在子项目下新增第二份依赖文件。
