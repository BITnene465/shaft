# Shaft 脚本使用说明

本文档说明仓库 `scripts/` 目录下的正式脚本如何使用。

范围约束：
- 只覆盖正式脚本入口
- 覆盖 `scripts/tasks/` 下的任务脚本
- **不覆盖 `scripts/tmp/`**；`tmp` 目录视为临时实验区，不属于稳定接口

## 1. 设计原则

`scripts/*.py` 的定位是**薄入口**：
- CLI 解析与命令编排放在 `src/shaft/cli`
- `scripts/*.py` 只负责调用对应 CLI 主入口

当前唯一例外是：
- `scripts/tasks/convert_grounding_structured_to_sft.py`
- `scripts/tasks/convert_grounding_structured_to_sft_row_major.py`

它属于明确的任务数据准备脚本，不是训练内核入口。

## 2. 顶层脚本

### `scripts/train.py`

用途：
- 统一训练入口
- 当前支持 `sft` 与 `rlhf` 子命令

常用形式：

```bash
python scripts/train.py sft --config configs/train/train_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/train_dpo_4b.yaml
```

兼容写法：

```bash
python scripts/train.py --config configs/train/train_sft_4b.yaml
```

说明：
- 如果直接传 `--config`，当前默认走 `sft`
- 真正的命令定义在 `src/shaft/cli`

### `scripts/infer.py`

用途：
- 运行可配置的多阶段推理 pipeline

常用形式：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png
```

带初始上下文：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png \
  --inputs '{"document_id":"demo-001"}'
```

说明：
- `--inputs` 是 JSON 字符串
- 输出会打印为 JSON

### `scripts/export.py`

用途：
- HF 兼容导出工具
- checkpoint 布局检查
- PEFT adapter 合并

子命令：
- `inspect`
- `validate`
- `merge-peft`

示例：

```bash
python scripts/export.py inspect --path outputs/run_x/checkpoint-100
```

```bash
python scripts/export.py validate \
  --path outputs/run_x/checkpoint-100 \
  --finetune-mode lora \
  --model-type qwen3vl
```

```bash
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path outputs/run_x/checkpoint-100 \
  --output-dir outputs/run_x/merged
```

### `scripts/web.py`

用途：
- 启动面向工程师/科研人员的 Web UI

常用形式：

```bash
python scripts/web.py
```

指定 host / port：

```bash
python scripts/web.py --host 0.0.0.0 --port 7861
```

指定默认训练配置：

```bash
python scripts/web.py --base-config configs/train/train_sft_4b.yaml
```

说明：
- 默认端口不固定；省略 `--port` 时由 Web UI 服务自动选择空闲端口
- `Ctrl-C` 视为正常退出

### `scripts/eval_bench.py`

用途：
- Eval Bench 的薄入口
- 进入 `projects/eval_bench` 子项目 CLI
- 创建 benchmark、初始化 run、校验和管理离线评测运行产物

环境：

```bash
uv pip install -e ".[dev,eval-bench]"
```

常用形式：

```bash
python scripts/eval_bench.py --help
```

从 `raw_data` 验证集复制一份 benchmark 到 `eval_bench_store/`：

```bash
python scripts/eval_bench.py create-benchmark \
  --benchmark-id multitask_val_v1 \
  --task detection \
  --task keypoint \
  --source-root data/raw_data \
  --source-manifest data/raw_data/splits/layout_val.txt \
  --split val \
  --layer layout \
  --layer arrow
```

```bash
python scripts/eval_bench.py init-run \
  --run-id demo \
  --model-id outputs/qwen3vl-sft/demo/best \
  --model-path outputs/qwen3vl-sft/demo/best \
  --benchmark-id multitask_val_v1 \
  --task detection \
  --benchmark-root eval_bench_store/benchmarks/multitask_val_v1/data \
  --benchmark-manifest eval_bench_store/benchmarks/multitask_val_v1/splits/val.txt \
  --benchmark-task detection \
  --benchmark-task keypoint \
  --split val \
  --spec-id grounding_layout.latest \
  --prompt-id grounding_layout.v1
```

```bash
python scripts/eval_bench.py validate-prediction \
  eval_bench_store/runs/demo/predictions/part2/json/pic001.json \
  --task detection
```

创建和查看持久化 eval job：

```bash
python scripts/eval_bench.py create-job \
  --kind eval \
  --payload-json '{"manifest":{"kind":"eval_job","runtime":{"mode":"ephemeral","engine":"vllm_openai","env":{"CUDA_VISIBLE_DEVICES":"0","CUDA_DEVICE_ORDER":"PCI_BUS_ID"},"args":{"model":"outputs/qwen3vl-sft/run/best","served-model-name":"qwen3vl-best","host":"127.0.0.1","port":8000,"tensor-parallel-size":1,"max-model-len":32768,"gpu-memory-utilization":0.9,"max-num-seqs":8,"trust-remote-code":true}},"eval":{"model_id":"qwen3vl-best","benchmark_id":"multitask_val_v1","task":"detection","prompt_id":"grounding_arrow.latest","target_labels":["arrow"],"prompt_path":"configs/prompts/grounding_arrow.yaml","generation":{"max_tokens":4096,"temperature":0,"top_p":1},"data":{"max_pixels":1048576,"batch_size":1}}}}'

python scripts/eval_bench.py list-jobs
```

登记和查看 model service：

```bash
python scripts/eval_bench.py register-service \
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

python scripts/eval_bench.py list-services
python scripts/eval_bench.py show-service --service-id local-vllm-0
python scripts/eval_bench.py service-command --service-id local-vllm-0
```

本地 vLLM 服务也可以通过 CLI 启停：

```bash
python scripts/eval_bench.py start-service --service-id local-vllm-0
python scripts/eval_bench.py stop-service --service-id local-vllm-0
```

处理下一个 queued eval job。CLI 会同步执行完整 worker，适合终端和脚本；Dashboard 启动后会由
后台 orchestrator 自动调度 queued eval job，一般不需要人工调用：

```bash
python scripts/eval_bench.py process-next-job
```

对已有 prediction snapshot 的 run 生成指标报告：

```bash
python scripts/eval_bench.py evaluate-run --run-id <run_id>
```

对两个已经 evaluate 的 run 生成 pairwise comparison report：

```bash
python scripts/eval_bench.py compare-runs \
  --baseline-run-id <old_run_id> \
  --candidate-run-id <new_run_id>
```

运行 dashboard store 的轻量性能 smoke：

```bash
python scripts/eval_bench.py perf-smoke --iterations 5 --sample-limit 500
```

启动 dashboard：

```bash
python scripts/eval_bench.py serve-dashboard --host 127.0.0.1 --port 8765
```

前端构建与渲染检查：

```bash
cd projects/eval_bench/frontend
npm install
npm run build
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run render-check
EVAL_BENCH_URL=http://127.0.0.1:8765/runs/<run_id>?sample=0 \
  INTERACTION_SMOKE=1 \
npm run render-check
npm run test:status-model
npm run test:manifest-tools
npm run test:workspace-settings
EVAL_BENCH_URL=http://127.0.0.1:8765 npm run test:dialogs
EVAL_BENCH_URL=http://127.0.0.1:8765/runs/<run_id> npm run test:viewer-performance
EVAL_BENCH_URL=http://127.0.0.1:8765/settings npm run test:settings-preview
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:shortcuts
EVAL_BENCH_URL=http://127.0.0.1:8765/ npm run test:layout
```

`test:layout` 会遍历核心页面和弹窗，在 desktop / compact / narrow 视口下检查全局滚动、局部滚动容器、
高级检索面板、独立 rank-board / compare chunk，并固定 Overview 的高价值约束：顶栏 status 必须是独立
圆角 capsule，Overview 必须保留 v11 flight deck surface：priority stage、command rail、hero next action、四段管线、行动入口、一个 operations surface 和最近 run 紧凑 ticker，
旧活动矩阵、mini chart wall 和 chart matrix 不能回流。Overview 不能出现 precision/recall/IoU
这类细指标文案，也不能回流 Notes、Label footprint、模型分布、Job 日历或 Scheduler 资源这类低价值总览面板；
flight deck 需要滚动时不能被 hidden 裁切，priority stage、command rail、operations、行动入口和最近 run 面板必须保持可读高度，不能在
compact / narrow 视口塌缩成 30-40px 外壳；最近 run 只能是可点击紧凑摘要，关键入口必须保留 hover/transition 反馈。
Benchmark / Run 检查器还会模拟样本过滤 0 命中，确认过滤入口、样本列表和主画布空状态留在同一个
inspector split 内，不能卸载成全页 EmptyState。
Runs / Rank Board 会实际输入检索词，先点击 `AdvancedFilterBar` 的条件 token 清除单个检索条件，再点击统一清空动作，
确认 search/number/select 按默认值恢复，且排序 select 不会被误算成生效过滤条件。高级检索浮层必须保持在同一工作区内滚动，
不能通过展开表单把主表格和核心面板挤出可视区域。
Rank Board 还会检查 facet rail 完整暴露 Tasks、Benchmarks、Status、Labels、Models、Prompts 和
Metrics 七类 `.rank-facet-button` 可点击 chip，点击后能把同一份高级检索状态置为 active，
长 facet 组必须提供 `.rank-facet-toggle` 展开/收起入口，防止核心排行榜退回静态 facet 摘要、
漏接后端 facet 或只渲染前 5 个值导致长尾筛选不可达。
UI contract 还会锁住 Rank Board 表格第一分数列必须使用 active primary metric label / score，并保留
leader-relative `score_delta` 展示，避免用户切换主指标后表格仍固定展示 F1 或只能看到孤立分数。

说明：
- `eval_bench` 是仓库内子项目，核心代码在 `projects/eval_bench/eval_bench`
- dashboard 前端在 `projects/eval_bench/frontend`，使用 React、Vite 和 TanStack
- dashboard 前端模块边界：`main.tsx` 只做路由和页面装配；dashboard state query 在 `dashboardState.ts`；业务状态在 `statusModel.ts`；浏览器设置和快捷键 action registry 在 `workspaceSettings.ts`；workspace split layout 在 `workspaceLayout.tsx`；总控工作台页面在 `overviewPage.tsx`；基准集页面和基准集真值检查器在 `benchmarksPage.tsx`；结果库、导入预测和 run 检查器在 `runsPage.tsx`；评测中心和 job queue 在 `jobsPage.tsx`；benchmark/run 表格在 `runTables.tsx`；复用过滤控件在 `filterControls.tsx`；共享样本分页在 `samplePager.tsx`；Run/Compare 共享样本叠图在 `sampleViewer.tsx`；成对样本对比详情在 `comparisonSamplePage.tsx`；基础输入控件在 `controlPrimitives.tsx`；viewer 渲染在 `viewerCanvas.tsx`；viewer 控制/对象面板在 `viewerPanels.tsx`；viewer 纯几何计算在 `viewerGeometry.ts`；metric 中间层在 `viewerMetrics.ts`；设置页页面在 `settingsPage.tsx`；设置页分组控件在 `settingsControls.tsx`；服务页在 `servicesPage.tsx`；manifest/prompt 转换在 `manifestTools.ts`；样本 URL、分页 offset 和 offset 合法化在 `sampleNavigation.ts`；业务 PNG 图标映射在 `iconLibrary.tsx`；backend/job log tail 共享逻辑在 `log_utils.py`
- dashboard 共享交互原语位于 `ui.tsx`：标准命令使用 `ActionButton` / `CommandButton` /
  `IconActionButton`，样本列表行使用 `SelectableRowButton`，query/label chip 使用
  `OptionChipButton`，对象行和可选卡片使用 `SelectableCardButton`，折叠面板入口使用
  `PanelToggleButton`，业务页和 viewer details/summary 折叠面板使用 `DisclosurePanel`；紧凑 select、表单 select、number、color 和 toggle 控件位于 `controlPrimitives.tsx`。
  `test:ui-contracts` 会阻止已收敛的 row/chip/select/submit 控件回流到业务页 raw class 拼接，并固定
  高级检索折叠头、条件 token 和清空动作必须在 `AdvancedFilterBar` 内统一实现；Overview 静态契约必须使用
  v11 flight deck surface 的 `overview-home-v11` / `overview-command-center` / `overview-priority-stage` /
  `overview-command-rail` / `overview-workbench` / `overview-ops-surface`，不能回流旧 `overview-home-v6`、
  `overview-home-v7`、`overview-home-v10`、`overview-command-deck`、`overview-focus-panel`、阻塞优先级面板、orbit 装饰或活动矩阵组件；
  首页和共享 workspace 入口必须保留 hover、active、pulse、radar/rail 这类触觉反馈，但不能新增 UI 私有状态真源。
- 依赖由仓库根目录 `pyproject.toml` 的 `eval-bench` extra 统一管理
- `scripts/eval_bench.py` 只负责把子项目加入 `sys.path` 并调用 CLI
- Eval Bench 自己管理 benchmark 数据；run 不直接读取训练 raw_data
- 默认持久化目录是 `eval_bench_store/`，不写入训练产物目录 `outputs/`
- job registry 使用 `eval_bench_store/db/eval_bench.sqlite`
- Dashboard 启动时会启动 Eval Bench orchestrator，自动扫描 queued eval job；它会根据 live running job 数、`cuda_visible_devices`、ephemeral runtime 端口和 `tensor_parallel_size` 判断资源是否足够，资源不冲突的 job 可并发启动。已请求取消但 worker/runtime 仍存活的 job 仍由 `job_lifecycle.py` 视为资源占用。默认并发上限为 2，可通过 `EVAL_BENCH_SCHEDULER_MAX_CONCURRENT_JOBS` 调整；扫描间隔可通过 `EVAL_BENCH_SCHEDULER_INTERVAL_S` 调整
- 当前 worker 支持 manifest-driven `eval_job`：claim queued job 后解析 manifest，按 `runtime.mode` 启动一次性 vLLM runtime 或连接已有 service，再写入 `runs/<run_id>/run.json`
- job 创建支持模板 + 自由 JSON manifest。默认 `eval_job` 模板是箭头检测，layout 检测保留为 `layout_eval_job`，箭头关键点评估保留为 `keypoint_eval_job`。Dashboard 的 Jobs 页提供 `Validate` preflight，会检查 benchmark/model/task/prompt/target labels，展示 vLLM 启动命令，并把未知 `runtime.args` 保留为 CLI flags
- CLI 和 Dashboard 的 `create-job` / `/api/jobs` 必须复用同一套 preflight；非阻塞 warning 会写入 job metadata 的
  `preflight_warnings`，方便 agent 后续通过 `list-jobs` / `dashboard-state` 排查风险。
- `list-agent-commands` 是 agent 稳定命令面的发现入口；run 初始化、prediction 文档校验和手动推进队列
  分别通过 `init-run`、`validate-prediction` 和 `process-next-job` 暴露，新增 agent 命令必须同步
  `AGENT_COMMAND_METADATA`、handler 映射和 CLI contract 测试。删除、归档、取消、停止这类危险生命周期命令
  还必须进入 `AGENT_DESTRUCTIVE_COMMANDS`，让 `list-agent-commands` 输出 `destructive=true`。
- job lifecycle 的 running 资源检查和 queued FIFO scan 必须使用 `EvalBenchDatabase.matching_jobs()` 的完整匹配集合；
  `list-jobs` / `/api/jobs` 的 `offset/limit` 分页只服务目录浏览，不能作为调度窗口。
- Dashboard 主页面不再使用嵌套 tab 承载低频表单；新建评测、创建 benchmark、导入 prediction snapshot 和登记 service 都通过临时弹层打开，主页面只保留队列、目录、结果和服务状态。
- Dashboard 业务图标库位于 `projects/eval_bench/frontend/public/icons/eval-bench/`，由 image_gen 母版裁剪得到；运行时统一通过 `iconLibrary.tsx` 使用，通用工具动作仍保留矢量图标。
- prompt template registry 的 repo 内置项会随代码启动刷新；用户从 dashboard 保存过的自定义 prompt 不会被内置 seed 覆盖。应用 detection prompt 时会同步写入或清空 `target_labels`，避免从 layout prompt 切换到 arrow prompt 后仍沿用 `icon/image/shape`；应用 keypoint prompt 或手动切到非 detection task 时，`manifestTools.ts` 会清理隐藏残留的 `target_labels` / `target_labels_source`，让后端 keypoint 默认策略固定解析为 `arrow`
- evaluator / import / comparison 统一通过 `eval_semantics.py` 解析评估语义。目标标签优先级是 run spec 显式 `target_labels`、prompt metadata、legacy prompt ID 兼容推断、task default、unscoped；report 会记录 `target_labels_source`。layout 检测只评价 `icon/image/shape`，arrow 检测只评价 `arrow`。keypoint 只允许 `arrow` 关键点评估范围，preflight / init-run / import / worker / evaluator 都会拒绝非 `arrow` 的显式 keypoint `target_labels`。导入外部 prediction snapshot 时也要传 prompt ID 或 `--target-label`，不能只用 `task=detection` 表示子任务；Dashboard 的 detection label 子任务面板只允许从 benchmark/prompt/manifest 候选 chip 中选择，不提供自由文本 label 添加入口。
- run manifest 会持久化模型路径、prompt ID/path/hash、prompt 文本快照、采样参数、pixel budget、job manifest 和 vLLM runtime/service 参数；dashboard 的 Run Inspector 顶部会按需展开这些配置
- sample image API 暴露三层图像资源：`/image` 返回原图，`/image/preview?max_side=1800` 返回缓存 JPEG 缩略代理，`/image/tiles/{level}/{x}/{y}` 返回缓存 JPEG 金字塔瓦片；dashboard viewer 默认使用 `image_preview_url`，高倍缩放停顿后延迟加载少量瓦片增强局部细节，派生缓存写入 `eval_bench_store/cache/image_proxy/`
- `register-service` 会把外部 vLLM endpoint 或本地 vLLM OpenAI server 参数写入 SQLite；dashboard 的 Services 页也使用同一套 API。长期 vLLM 放在 Services 页管理，一次性 vLLM 放在 job manifest 的 `runtime.mode=ephemeral` 中管理
- `start-service` 使用当前 `.venv` 的 Python 启动 `vllm.entrypoints.openai.api_server`，日志写入 `eval_bench_store/services/<service_id>/service.log`；`stop-service` 会终止该本地服务进程
- job manifest 使用 `runtime.mode=ephemeral` 时，worker 会启动 job 专属 vLLM，等待 endpoint ready，执行推理后关闭进程；runtime 日志写入 `runs/<run_id>/logs/runtime.log`，Dashboard 通过 `/api/jobs/<job_id>/logs` 读取 tail
- `/api/jobs/<job_id>/cancel` 可取消 queued job，也可终止 running job；running job 会写入 `cancel_requested`，worker 在 runtime 启动、样本推理和评估阶段检查该标记，并尽量终止 job 专属 ephemeral runtime 进程组。
- worker 执行时会持续更新 job metadata 中的 `progress_phase`、`progress_done`、`progress_total`、`progress_current_sample` 和 `progress_message`；Dashboard 的任务中心每 2 秒刷新 job 和 scheduler 状态。runtime log 不在主列表常驻展示，点击某条 job row 后才会在嵌套详情面板中读取 `/api/jobs/<job_id>/logs?max_lines=0` 的完整日志
- job manifest 使用 `runtime.mode=existing_service` 且提供 `endpoint` 时，worker 会调用 OpenAI-compatible `/v1/chat/completions`，写入 raw output、prediction snapshot 和 metric report，但不负责启停该服务
- job payload 中设置 `"backend":"dry_run"` 时，会写空 prediction snapshot 并生成报告，用于端到端链路测试
- `evaluate-run` 消费已经存在的 prediction snapshot，输出 `runs/<run_id>/reports/metrics.json` 和轻量 `runs/<run_id>/reports/summary.json`；前者包含整体指标、per-label 指标和 sample-level TP/FP/FN 诊断，后者供 dashboard 高频列表刷新使用
- 预测结果和 test/GT 的对比不直接读取训练目录：先把 test split 复制成 benchmark，再把预测结果写入 `runs/<run_id>/predictions/`，最后由 `evaluate-run` 读取 benchmark GT 和 prediction snapshot 计算 TP/FP/FN、IoU 和 per-label 指标；Run Inspector 展示逐图 GT / Prediction 叠图
- 如果预测 JSON 已经在外部目录，可以用 `import-predictions --benchmark-id <benchmark> --prediction-root <dir> --run-id <run>` 导入为 Eval Bench run；它会按 benchmark split 的相对路径、image path 或 basename 对齐 prediction JSON，默认导入后立即运行 `evaluate-run`。Dashboard 的 Runs 页也提供默认折叠的 `Import prediction snapshot` 入口，提交后直接生成可打开的 run
- `resolve-target-labels` 是 agent 创建 detection label 子任务前的稳定查询入口；它复用 benchmark summary、
  prompt template metadata 和 `label_policy.py`，返回最终 target labels、候选 labels、来源和拼写校验结果。
- `show-job-template`、`show-prompt-template`、`show-job`、`show-benchmark` 和 `show-service`
  是 agent 读取单个模板、job、benchmark summary 与 model service 详情的稳定 CLI；不要求 agent
  扫描 benchmark artifact、读取 SQLite 或依赖 dashboard 前端 state。
- `compare-runs` 消费两个已有 `metrics.json`，输出 `eval_bench_store/exports/comparisons/<baseline>__vs__<candidate>.json`，其中包含整体指标 delta、样本改善/退化统计、sample index 和 top case 列表
- `show-comparison` 和 `show-comparison-sample` 是 agent 读取已保存 comparison 和成对样本详情的稳定 CLI；
  不要求 agent 直接读取 `exports/comparisons` 或 run artifact 文件。
- CLI 子命令统一通过 `eval_bench.cli._command_handlers()` 分发；新增子命令必须同时进入 parser 和
  handler 映射，`projects/eval_bench/tests/test_cli.py` 会检查 parser 命令集合与 handler 集合一致，并锁住
  `AGENT_COMMAND_METADATA` / `AGENT_STABLE_COMMANDS`。agent 可用 `list-agent-commands` 发现当前稳定命令面，
  并读取每条命令的 `domain`、`mutates_state`、`arguments` 和 `mutually_exclusive_groups` 来区分只读查询、
  artifact 写入、service/job 生命周期操作和参数形态。
- `perf-smoke` 对 state summary、saved comparison listing、benchmark sample listing、run sample listing 等 dashboard 常用路径做本地耗时测量，输出 JSON 摘要
- dashboard 的 Benchmarks 页可以从 raw_data split 创建 benchmark copy；benchmark inspector 会按样本读取 benchmark GT，并在原图上叠加 GT 框和 keypoints，支持服务端分页、sample 序号跳转和按 label 过滤样本
- dashboard 的 run inspector 会按样本读取 benchmark GT 与 prediction snapshot，并在原图上叠加 GT / Prediction 框、linestrip 和 keypoints，支持服务端分页以及按 error type / label 过滤样本，用于快速定位漏检、误检和解析问题
- sample viewer 支持 label 过滤、sample 分页浏览、GT/Prediction 与 box/line/point 图层显隐、可见标签 metric 聚合、对象列表联动、hover/click 高亮、滚轮缩放、拖拽平移和可配置快捷键；Run Inspector 以图像画布为主区域，配置、prompt 和 label metric 明细默认折叠；`INTERACTION_SMOKE=1 npm run render-check` 会自动跑一次 sample 切换、筛选、复选框、缩放、平移和对象点击 smoke；`npm run test:shortcuts` 会静态扫描键盘入口，并在 benchmark、run、compare、settings 页面验证 keymap 行为
- dashboard 的 Compare 页提供基于 report summary 的排行榜、task/benchmark 过滤、成对 run 选择、整体 delta、已保存 comparison 列表，以及可跳转到并排样本对比视图的 top 改善/退化样本列表

## 3. `scripts/tasks/`

`scripts/tasks/` 用于**任务级数据准备或转换脚本**。

这类脚本可以：
- 读写数据文件
- 生成训练前产物
- 服务具体业务任务

但不应：
- 承载训练内核语义
- 复制一套新的训练 CLI
- 替代 `src/shaft/cli`

### `scripts/tasks/convert_grounding_structured_to_sft.py`

用途：
- 把 `structured/*.jsonl` 的 grounding 结构化 GT 转成当前框架可训练的 `jsonl_sft`

输入要求：
- 结构化 GT 至少包含：
  - `sample_id`
  - `image_path`
  - `image_width`
  - `image_height`
  - `instances`
- `instances` 中每个元素至少有：
  - `label`
  - `bbox`

输出字段：
- `image_path`
- `sample_id`
- `dataset_name`
- `system_prompt`
- `user_prompt`
- `target_text`
- `extra`

关键行为：
- `bbox` 会量化到 `1000` bins，输出为 `bbox_2d`
- `target_text` 是纯 JSON array
- 排序规则：
  - 先按 `bbox_area / image_area` 的 **log 尺度分桶**
  - 当前默认 `bucket_base = 1.5`
  - 同桶内按 `(y1, x1, y2, x2, label)` 排序
- prompt 从 YAML 配置文件读取

常用形式：

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/train.jsonl \
  --output data/grounding_arrow/sft/train.jsonl \
  --dataset-name grounding_arrow
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/val.jsonl \
  --output data/grounding_arrow/sft/val.jsonl \
  --dataset-name grounding_arrow
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow_syn/structured/train.jsonl \
  --output data/grounding_arrow_syn/sft/train.jsonl \
  --dataset-name grounding_arrow_syn
```

常用参数：
- `--prompt-config`
  - 默认：`configs/prompts/grounding_arrow.yaml`
- `--num-bins`
  - 默认：`1000`
- `--area-bucket-base`
  - 默认：`1.5`
- `--no-readme`
  - 跳过输出目录下的 `README.md`

示例：

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/train.jsonl \
  --output data/grounding_arrow/sft/train.jsonl \
  --dataset-name grounding_arrow \
  --prompt-config configs/prompts/grounding_arrow.yaml \
  --num-bins 1000 \
  --area-bucket-base 1.5
```

### `scripts/tasks/convert_grounding_structured_to_sft_row_major.py`

用途：
- 把 `structured/*.jsonl` 的 grounding 结构化 GT 转成当前框架可训练的 `jsonl_sft`
- 使用 `row-major` 的 canonical order，而不是面积分桶优先

关键行为：
- `bbox` 会量化到 `1000` bins，输出为 `bbox_2d`
- `target_text` 是纯 JSON array
- 排序规则：
  - 先基于量化后的 `bbox_2d` 计算 `y_center`
  - 用 `row_bucket = floor(y_center / row_bucket_size)` 做视觉行分组
  - `row_bucket_size = max(8, round(median_height * 0.5))`
  - 桶内按 `(x1, y1, y2, x2, label)` 排序
  - 若量化坐标相同，再用原始浮点 bbox 做 tie-break，保证稳定性
- prompt 从 YAML 配置文件读取

常用形式：

```bash
python scripts/tasks/convert_grounding_structured_to_sft_row_major.py \
  --input data/grounding_arrow/structured/train.jsonl \
  --output data/grounding_arrow/sft/train.jsonl \
  --dataset-name grounding_arrow
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft_row_major.py \
  --input data/grounding_layout/structured/train.jsonl \
  --output data/grounding_layout/sft/train.jsonl \
  --dataset-name grounding_layout \
  --prompt-config configs/prompts/grounding_layout.yaml
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft_row_major.py \
  --input data/grounding_arrow_syn/structured/train.jsonl \
  --output data/grounding_arrow_syn/sft/train.jsonl \
  --dataset-name grounding_arrow_syn_avg
```

常用参数：
- `--prompt-config`
  - 默认：`configs/prompts/grounding_arrow.yaml`
- `--num-bins`
  - 默认：`1000`
- `--no-readme`
  - 跳过输出目录下的 `README.md`

示例：

```bash
python scripts/tasks/convert_grounding_structured_to_sft_row_major.py \
  --input data/grounding_layout/structured/val.jsonl \
  --output data/grounding_layout/sft/val.jsonl \
  --dataset-name grounding_layout \
  --prompt-config configs/prompts/grounding_layout.yaml \
  --num-bins 1000
```

原子写入约束：
- JSONL 输出使用临时文件写完后 `os.replace`
- README 也使用原子替换
- 目的是避免训练进程读到半成品


## 4. 维护规则

新增脚本时，至少需要同步更新本文件，说明：
- 脚本用途
- 输入输出
- 关键参数
- 示例命令

如果脚本只是一次性临时实验，不应写进这里，而应留在 `scripts/tmp/`。
