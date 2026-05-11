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
- UI 基础组件：Radix UI
- 前端 metric 中间层：`projects/eval_bench/frontend/src/viewerMetrics.ts`

FastAPI app 同时提供 `/api/*` 和构建后的 Vite SPA。本地前端开发时，Vite 会把 `/api` 代理到 dashboard API。

## Dashboard 页面约定

- `评测中心` 是日常入口：同页展示短生命周期 job 队列、调度状态、runtime log 和最近 run 结果。
  job 成功后会在详情里直接链接到对应 run；完整历史仍可在 `结果库` 中查询、删除、归档和重新评估。
- `结果库` 面向已落盘 run snapshot；它不是另一个任务中心，只管理可复查的预测、报告和导入结果。
- `基准集检查` 和 run 检查器共用同一套 `CanvasStage`，画布区域固定在视口内，支持滚轮缩放、拖拽平移、
  label 过滤、对象 hover/active 高亮和样本级预取。
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

Eval Bench 按内部任务平台拆成三层：

- Control Plane：Dashboard API、SQLite registry、Job manifest、Service registry、preflight 校验和状态机。
- Execution Plane：worker 领取 job，按 manifest 启动一次性 runtime 或连接已有 service，执行推理、解析、评估和 artifact 落盘。
- Artifact Plane：`eval_bench_store/` 中的 benchmark copy、run manifest、prediction snapshot、raw output、runtime log、report、comparison 和 trash。

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

Jobs 页可以选择 prompt template，并把 system prompt、user prompt、parser、metric profile、generation 和 data 默认值写回 manifest。用户也可以直接编辑 manifest 中的 prompt 字段，并把当前 manifest 的 prompt 保存为新的模板。后端 preflight 会用同一份 prompt registry 解析模板；job 入队时保存 resolved manifest，worker 后续执行不依赖前端临时状态。

每个 prompt template 同时声明 `target_labels`，这是评测语义的一部分，不是纯展示字段。多任务
benchmark 里可能同时有 layout 和 arrow 标注，但一次 run 只应该评价当前 prompt 要求模型输出的
label 集合：

- `grounding_layout.latest` 只评价 `icon / image / shape`。
- `grounding_arrow.latest` 只评价 `arrow`。
- `keypoint_arrow.latest` 只评价 `arrow`，并额外评价关键点。

Evaluator 会在匹配前按 `target_labels` 同时过滤 GT 和 prediction，report summary 和 comparison
也会记录这组 label。这样 layout prompt 在包含 arrow GT 的 benchmark 上评测时，不会把未要求输出的
arrow 误算成漏检；如果要做全任务评测，应显式使用包含所有目标 label 的 prompt/template，而不是只改
`task=detection`。

`eval_job` 的最小 manifest 形态：

```json
{
  "kind": "eval_job",
  "runtime": {
    "mode": "ephemeral",
    "engine": "vllm_openai",
    "env": {"CUDA_VISIBLE_DEVICES": "0,2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    "args": {
      "model": "outputs/qwen3vl-sft/4b/run/best",
      "served-model-name": "qwen3vl-latest",
      "host": "127.0.0.1",
      "port": 8000,
      "tensor-parallel-size": 2,
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
    "prompt_id": "grounding_layout.latest",
    "target_labels": ["icon", "image", "shape"],
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

`benchmark.json` 是 benchmark manifest。`run.json` 是 run manifest。每个 prediction JSON 都是带 sample-level metadata 的 raw-data-like prediction document。

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
- 用于创建 benchmark、初始化 run manifest、校验 prediction document、创建/查看 job、执行 job preflight、启动 dashboard、读取 backend log 的轻量 CLI/API
- 用于登记、查看、启动、停止、健康探测和日志 tail 的 service registry；也支持只登记外部 vLLM endpoint
- 将模型文本归一化为 prediction document 的 parser 工具，并在写入 snapshot 前按 same-label high-IoU 去重
- 从持久化 prediction snapshot 生成报告的 `evaluate-run`
- React dashboard 外壳：中文工程控制台、benchmark 创建、benchmark sample 浏览、run、持久化 queue 状态、manifest-driven job 创建、job preflight、外部 prediction snapshot 导入、run 评估、run prompt/推理配置查看、样本级 GT/prediction 检查，以及带 task/benchmark filter 的报告驱动 leaderboard；主要检查与对比工作区支持可拖拽分栏并持久化用户布局
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
  --prompt-id imported
```

`import-predictions` 会按 benchmark split 对齐预测文件，优先匹配同相对路径
（如 `part1/json/a.json`），其次匹配 image path 对应 JSON，最后匹配 basename。
默认允许缺失预测，并在 report 中记录 missing prediction；如果希望缺失时直接失败，
加 `--strict`。如果同一个 `run-id` 已存在，加 `--overwrite` 才会替换。

Dashboard 的 Runs 页也提供同一能力：展开 `Import prediction snapshot`，填写
`run_id`、benchmark、prediction root、task、model ID 后提交。后端会创建标准 run、
复制 prediction snapshot、默认立即评估，之后直接进入 Run Inspector 看逐图 GT / Prediction
对比。

Dashboard 的 Benchmarks 页也可以创建 benchmark copy：展开 `Create benchmark copy`，
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

开启画布交互 smoke 后运行同一检查：

```bash
cd projects/eval_bench/frontend
EVAL_BENCH_URL=http://127.0.0.1:8765/runs/<run_id>?sample=0 \
  INTERACTION_SMOKE=1 \
  npm run render-check
```

`INTERACTION_SMOKE=1` 会实际操作查询筛选、标签 / 图层开关、滚轮缩放、画布拖拽、对象列表点击和基础快捷键。评测检查器的交互设计以图像工作台为主：应用固定在一个屏幕内，页面本身不滚动；样本列表、对象列表和表格只在自己的面板内滚动；配置、prompt 和标签指标明细默认折叠，低频的创建/注册表单不抢占主画布。前端 metric 中间层还有独立检查：

```bash
cd projects/eval_bench/frontend
npm run test:metrics
```

Dashboard v2 的基本操作约定：

- 评测记录页是 run 管理中心：支持 run、模型、基准集搜索，支持状态、任务、基准集筛选，两条 run 勾选后直接进入对比页，也可以重新评估、归档和删除。
- 任务中心页是队列管理中心：Dashboard 后台 orchestrator 会自动推进 queued eval job；表格展示 scheduler 状态、实时阶段、进度条和当前 sample。runtime log 不常驻占用列表空间，点击某条 job row 后会在嵌套详情面板中查看完整日志。排队中的 job 可以取消；无效、失败或 demo job 可以删除。删除记录会写入 `eval_bench_store/trash/`，默认不硬删证据。
- 模型服务页管理模型服务 registry：本地服务可启动/停止，服务记录可删除；删除前会先尝试停止运行中的本地服务。
- 工作台设置页维护全局用户偏好：状态颜色、按 label 的实例颜色、框线宽、骨架线宽、点半径、标签字号、描边、透明度、预测线型、滚轮缩放灵敏度、拖拽平移灵敏度和缩放上下限会写入浏览器本地设置，所有检查器自动读取；页面内的叠图预览使用 `projects/eval_bench/static/settings_preview.svg` 作为稳定样例图，并叠加固定 GT/Pred demo instances。
- 左侧导航栏支持通过顶部图标按钮收起/展开；折叠状态也写入浏览器本地设置，便于大屏评测时把主画布面积留给工作区。
- 评测检查器快捷键：滚轮缩放，拖拽平移，`[` / `]` 切换样本，`Esc` 清除选中，`F` 复位视图，`G`/`P` 显隐真值/预测，`B`/`L`/`K` 显隐框/线/点。

## Dashboard 工作区交互规范

Eval Bench 的 dashboard 按工程工作台设计，不按长页面表单设计。页面主区域应该尽量固定在一个屏幕内，
低频配置折叠到详情面板，用户主要通过画布、列表和表格完成排障。

当前检查器工作区遵循这些约定：

- 基准集检查、run 检查和工作台设置页的叠图预览复用同一个 `CanvasStage`，不要再各自实现独立画布。
- 图片在容器内按原始宽高比自适应显示，默认完整可见；缩放范围由 `workspaceSettings.ts` 统一管理，默认 25% 到 800%，可在工作台设置中调整。
- 鼠标滚轮缩放是连续缩放；左键拖拽平移。滚轮和平移灵敏度由工作台设置统一管理，缩小到 100% 以下时也允许拖拽，便于用户调整画布位置。
- 样本列表 / 主画布、主画布 / 对象检查器、Compare 工作区、成对样本对比、Settings 外观页和 Job manifest 编辑/预检查区都使用可拖拽分栏，拖拽后的宽度写入浏览器本地设置；分栏范围应给足，不要只允许小范围微调。
- 样本列表会预加载当前样本前后若干张图，避免翻页或键盘切换时出现明显空白等待。
- 样本卡片、对象列表和诊断文本默认允许换行；不能用省略号隐藏 image path、label、IoU 或 bbox 等排障信息。
- 样本跳转输入框已取消。按样本浏览使用列表、分页和 `[` / `]`；精确定位后续应通过查询/过滤能力完成，而不是在侧栏常驻一个窄输入框。
- 表格行选择使用显式 checkbox 样式，不依赖浏览器默认外观；选择列不应出现无语义装饰点。

叠图颜色分两层：

- 状态颜色：`fn`、`fp` 和 active highlight 优先表达漏检、误检和当前高亮。
- Label 颜色：正常匹配或未参与匹配的实例按 label 着色；label 名不在代码中预设，用户可以在设置页手动添加任意 label 颜色规则，运行时按实际 label 匹配。

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

创建并查看持久化 eval job。Dashboard 推荐路径是在任务中心页展开“新建评测任务”，选择模板后直接编辑
manifest，先点“预检查”查看 preflight，再提交。CLI 仍然支持直接提交 JSON：

```bash
.venv/bin/python scripts/eval_bench.py create-job \
  --kind eval \
  --payload-json '{"manifest":{"kind":"eval_job","runtime":{"mode":"ephemeral","engine":"vllm_openai","env":{"CUDA_VISIBLE_DEVICES":"0,2","CUDA_DEVICE_ORDER":"PCI_BUS_ID"},"args":{"model":"outputs/qwen3vl-sft/run/best","served-model-name":"qwen3vl-best","host":"127.0.0.1","port":8000,"tensor-parallel-size":2,"max-model-len":32768,"gpu-memory-utilization":0.9,"max-num-seqs":8,"trust-remote-code":true}},"eval":{"model_id":"qwen3vl-best","benchmark_id":"multitask_val_v1","task":"detection","prompt_id":"grounding_layout.latest","target_labels":["icon","image","shape"],"prompt_path":"configs/prompts/grounding_layout.yaml","generation":{"max_tokens":4096,"temperature":0,"top_p":1},"data":{"max_pixels":1048576,"batch_size":1}}}}'

.venv/bin/python scripts/eval_bench.py list-jobs
```

登记本地 vLLM OpenAI server：

```bash
.venv/bin/python scripts/eval_bench.py register-service \
  --kind local_vllm \
  --service-id local-vllm-0 \
  --model-path outputs/qwen3vl-sft/run/best \
  --served-model-name qwen3vl-best \
  --cuda-visible-devices 0,2 \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16

.venv/bin/python scripts/eval_bench.py list-services
.venv/bin/python scripts/eval_bench.py service-command --service-id local-vllm-0
```

本地服务可以由 Eval Bench 启停。启动时使用当前 `.venv` 的 Python 执行 `python -m vllm.entrypoints.openai.api_server`，日志写入 `eval_bench_store/services/<service_id>/service.log`。外部服务只登记 endpoint，不由 Eval Bench 启停。

```bash
.venv/bin/python scripts/eval_bench.py start-service --service-id local-vllm-0
.venv/bin/python scripts/eval_bench.py service-health --service-id local-vllm-0
.venv/bin/python scripts/eval_bench.py service-logs --service-id local-vllm-0 --max-lines 200
.venv/bin/python scripts/eval_bench.py stop-service --service-id local-vllm-0
```

`start-service` 只负责拉起本地进程，状态会先进入 `starting`；`service-health` 会探测 OpenAI-compatible `/v1/models` endpoint，探测成功后状态变为 `running`，探测失败但本地进程还活时保持 `starting`，进程不存在则变为 `stopped`。Dashboard 的 Services 页复用同一套接口，显示 health、最近检查时间、错误信息和日志 tail。

处理下一个 queued eval job。CLI 会同步执行完整 worker；Dashboard API 则后台启动 worker 并立即返回
running job，供前端轮询状态和进度：

```bash
.venv/bin/python scripts/eval_bench.py process-next-job
```

第一个 worker action 有意命名为 `prepare_run`：它会 claim 一个 queued job，写入 `eval_bench_store/runs/<run_id>/run.json`，把 run manifest path 记录到 job metadata 中，并把真实模型推理留给后续 worker layer。

当 job manifest 使用 `runtime.mode=ephemeral` 且 `runtime.engine=vllm_openai` 时，worker 会先启动一个该 job 专属的 vLLM OpenAI server，等待 `/v1/models` ready，再执行推理，最后关闭这个进程。runtime stdout/stderr 写入 `runs/<run_id>/logs/runtime.log`，Dashboard 可通过 `/api/jobs/<job_id>/logs` 读取 tail。当 job manifest 使用 `runtime.mode=existing_service` 并提供 `endpoint` 时，worker 不负责启停模型服务，只通过 OpenAI-compatible chat-completions 请求执行推理。`endpoint` 可以是 server root、`/v1` 或 `/v1/chat/completions`。

做集成测试时，可以在 job payload 中设置 `"backend":"dry_run"`。worker 会为每个 benchmark sample 写入空 prediction snapshot，执行 `evaluate-run`，并把 run 标记为 `succeeded`。这可以在不调用模型后端的情况下验证完整 artifact 链路。

评估已有 run 的 prediction snapshot：

```bash
.venv/bin/python scripts/eval_bench.py evaluate-run --run-id <run_id>
```

第一份 metric report 会写入 `runs/<run_id>/reports/metrics.json`，内容包括 IoU@0.50 precision/recall、mean IoU、per-label count、缺失 prediction file，以及 keypoint 任务下的 endpoint distance。它也会保存样本级诊断信息：TP/FP/FN 数量、match pair、per-sample mean IoU、false positive reference 和 false negative reference。`evaluate-run` 还会写入 `runs/<run_id>/reports/summary.json`，这是不含 sample list 的小型 dashboard index 文件，避免频繁刷新的 run table 解析完整 diagnostic report。

Dashboard 的基准集检查器会直接读取 copied benchmark GT，并在 copied image 上叠加实例，支持服务端分页和标签过滤。评测检查器会读取同一份 benchmark GT 与 run prediction snapshot，叠加真值 / 预测的 bbox、linestrip 和 keypoint，并展示样本级诊断信息；它支持服务端分页、错误类型过滤和标签过滤，因此视觉排障不需要重新跑推理，也不需要手工打开生成文件。

Sample viewer 的目标更接近标注工具，而不是静态 preview：用户可以过滤 label，切换真值 / 预测以及框 / 线 / 点图层，调节状态颜色、任意 label 颜色规则、框线宽、骨架线宽、点半径、标签字号、高亮宽度、透明度、预测线型和鼠标交互参数，hover 或 click 某个 object 时会在对象列表和叠图中同步高亮，也可以使用滚轮缩放和拖拽平移查看局部。label 颜色由用户手动添加，运行时按实际 label 匹配，不预设 `arrow/icon/text` 这类固定任务名；linestrip 会绘制起点、终点和位于中间线段上的自适应方向三角形。Run inspection、工作台设置预览和成对样本对比复用同一个 viewer。工作台设置预览只替换底图和 demo instances，不另写独立可视化逻辑。原生 wheel listener 由 React effect 清理，sample list 走服务端分页并预加载当前样本附近的图片，以保证大 benchmark 下 dashboard 仍然流畅。

Metric 展示分三层：

- 总览：当前可见 label 下的 GT、Pred、TP、FP、FN 和 mean IoU。
- 分 label 明细：每个 label 的 GT、Pred、TP、FP、FN、P@0.50、R@0.50 和 mean IoU。
- 对象级诊断：每个 GT/Pred object 都显示 TP/FP/FN/unchecked 状态、匹配的对侧 object index、IoU 和 bbox 坐标。组件不直接推导这些状态，统一由 `viewerMetrics.ts` 计算，便于后续扩展新的任务类型、匹配策略和 object schema。

比较两个已经评估过的 run：

```bash
.venv/bin/python scripts/eval_bench.py compare-runs \
  --baseline-run-id <old_run_id> \
  --candidate-run-id <new_run_id>
```

Comparison report 会写入 `eval_bench_store/exports/comparisons/`。它只比较已经持久化的 metric report，不重新跑推理。Dashboard 的 Compare 页通过 `/api/comparisons` 读取同一份 report，展示 P/R/IoU 与 TP/FP/FN delta，列出已保存 comparison，并提供 top 改善/退化样本到并排样本对比 viewer 的跳转。Compare 工作区的 run rail、报告区、排行榜，以及成对样本对比的左右 run 面板都使用可拖拽分栏，适合在不同屏幕宽度下长期排障。

运行轻量 dashboard-store performance smoke：

```bash
.venv/bin/python scripts/eval_bench.py perf-smoke --iterations 5 --sample-limit 500
```

这个命令会测量 dashboard 常用路径：state summary、saved comparison listing、benchmark sample listing，以及存在 run 时的 run sample listing。它是本地迭代的 smoke signal，不替代更重的 load testing。

`scripts/eval_bench.py` 是薄包装入口，只负责把 `projects/eval_bench` 加入 `sys.path`，然后调用 `eval_bench.cli`。共享依赖继续维护在根目录 `eval-bench` extra 中，不要在子项目下新增第二份依赖文件。
