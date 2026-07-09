# Shaft 测试体系规范

本文档是 Shaft 测试边界、执行命令和重构方向的真源。目标不是追求测试数量，而是让测试稳定地证明框架行为，同时不把测试本身写成第二套业务实现。

## 1. 目标

- 默认测试必须快、稳定、可在普通开发环境运行。
- 重型训练、真实模型、真实服务和浏览器布局检查必须显式分层。
- 测试优先锁 public API、配置语义、artifact contract 和跨模块装配结果。
- 测试辅助代码集中到 `tests/support/` 或对应子项目的 `tests/support/`，不在每个测试文件里重复造半套运行时。
- 示例配置、脚本入口和真实数据文件只用于 contract / smoke / integration；普通单元测试不依赖仓库外部状态或运行产物。

## 2. 测试层级

### 2.1 Unit

标记：`unit`

用途：
- 纯函数、dataclass、registry、schema、codec、metric primitive。
- 不读真实仓库配置，不启动 subprocess，不加载真实模型，不依赖网络或 GPU。

允许：
- 内存对象。
- 很小的 `tmp_path` 文件，前提是被测对象本身就是文件解析器。

禁止：
- 读取 `configs/` 中的真实 YAML。
- patch 深层私有实现来绕开主流程。
- 启动 `scripts/*.py`。

### 2.2 Component

标记：`component`

用途：
- 通过模块 public API 验证一个组件或一条局部装配链。
- 例如 config loader、data center、trainer helper、export validator、webui controller。

允许：
- `tmp_path`。
- `tests/support` 中的 fake model、fake processor、config builder、artifact builder。
- 对外部边界做注入或 patch，例如 HTTP client、process launcher、model loader。

禁止：
- 把多个业务页面、CLI、数据库、worker 和真实 runtime 混成一个超长测试。
- 在测试文件内复制大量 YAML / JSON / manifest 模板；应抽到 builder。

### 2.3 Contract

标记：`contract`

用途：
- 验证 CLI/API/schema/manifest 的稳定输入输出。
- 证明用户或 agent 能依赖的接口没有漂移。

允许：
- 读取仓库内小型示例配置。
- 调用 parser / command handler / FastAPI route。
- 少量 subprocess `--help` 或 JSON error contract。

要求：
- contract 测试只断言稳定字段和行为，不断言实现内部调用顺序。
- 如果测试需要大量 store/run/benchmark 数据，应使用 `tests/support` 场景 builder。

### 2.4 Smoke

标记：`smoke`

用途：
- 跑最短可行主链，尽早暴露装配失败。
- 例如 CPU fake SFT、RLHF 最短链、torchrun DDP canary、online eval canary。

默认行为：
- `smoke` 不进入 `pytest -q`。
- 需要显式执行 `pytest -q -m smoke`。

要求：
- 使用最小 fake model / smoke model。
- 数据、配置和输出全部写入 `tmp_path`。
- 不依赖用户机器上的 `models/`、`outputs/`、`eval_bench_store/`。
- 如果依赖环境限制，例如 torchrun rendezvous，被阻断时必须清晰 skip。

### 2.5 Integration

标记：`integration`

用途：
- 真实模型、真实推理后端、真实外部服务、真实较长流程。

要求：
- 默认不运行。
- 模型路径、服务地址、GPU 条件都必须显式检测并 skip。

### 2.6 Manual

标记：`manual`

用途：
- 人工触发、耗时、重资源或环境绑定验证。

要求：
- 可以与 `integration` 或 `smoke` 叠加。
- 必须在测试内写清 skip 原因。

## 3. 执行命令

默认核心回归：

```bash
uv run pytest -q
```

主链 smoke：

```bash
uv run pytest -q -m smoke
```

真实集成：

```bash
uv run pytest -q -m integration
```

人工重型验证：

```bash
uv run pytest -q -m manual
```

Eval Bench 后端 focused 回归：

```bash
uv run pytest -q projects/eval_bench/tests -m "not smoke and not integration and not manual"
```

Eval Bench 前端按需人工执行，不作为训练框架 CI 门槛：

```bash
cd projects/eval_bench/frontend
npm run test:ui-contracts
npm run build
npm run test:layout
```

静态质量：

```bash
uv run ruff check src/shaft projects/eval_bench/eval_bench tests projects/eval_bench/tests
uv run python -m compileall src/shaft tests projects/eval_bench/eval_bench projects/eval_bench/tests
```

## 4. 测试支撑层

核心仓库使用：

- `tests/conftest.py`
  - 仓库通用 fixture。
  - 不放业务模板和复杂 fake。
- `tests/support/configs.py`
  - 最小 RuntimeConfig / YAML builder。
  - smoke 数据与配置生成。
- `tests/support/cli.py`
  - CLI Namespace builder、稳定 command invocation helper。
- `tests/support/training.py`
  - `TrainingArguments` builder、tiny model、trainer log capture、static online eval runner。
- `tests/support/pipeline.py`
  - pipeline 测试共享 fake model/tokenizer/processor/trainer、model artifact builder 和 SFT pipeline config builder。
- `tests/support/online_eval.py`
  - online eval runner 测试共享 fake trainer/model/collator、policy builder 和 batch builder。
- `tests/support/rlhf.py`
  - RLHF pipeline/smoke 测试共享 DPO/PPO/GRPO 最小 YAML 与 JSONL builder。

后续按需补充：

- `tests/support/fakes.py`
  - 共享 fake tokenizer、fake processor、fake trainer。
- `tests/support/artifacts.py`
  - 临时 HF / PEFT / checkpoint artifact builder。

Eval Bench 使用：

- `projects/eval_bench/tests/support/files.py`
  - JSON/file 写入等最小文件 helper。
- `projects/eval_bench/tests/support/cli_contracts.py`
  - CLI JSON schema contract 断言。
- `projects/eval_bench/tests/support/store.py`
  - 共享 benchmark/run/report store 场景。
- `projects/eval_bench/tests/support/jobs.py`
  - job manifest payload builder。
- `projects/eval_bench/tests/support/evaluator.py`
  - evaluator/run/comparison 测试共享最小 run manifest builder。

Eval Bench 后续按需补充：

- `projects/eval_bench/tests/support/runs.py`
- `projects/eval_bench/tests/support/benchmarks.py`
- `projects/eval_bench/tests/support/api.py`

规则：

- 测试文件只能组合 builder，不应维护大段重复 YAML / JSON。
- builder 必须生成最小有效对象，不应偷偷补入与测试无关的业务默认值。
- fake 应表达外部边界，例如 model loader、HTTP client、process launcher；不要 fake 被测模块内部算法。

## 5. 真实文件依赖规则

允许读取真实仓库文件的场景：

- `contract`：验证示例配置、CLI help、JSON schema、文档中承诺的入口。
- `integration/manual`：验证真实模型、真实服务、真实资源路径。

默认禁止：

- Unit / component 测试读取 `configs/train/*.yaml` 作为输入。
- 测试依赖 `outputs/`、`models/`、`temp/`、`eval_bench_store/`、`wandb/`。
- 测试从 `node_modules/`、`dist/` 或 `__pycache__/` 读取任何东西。

如果确实需要文件输入：

- 用 `tmp_path` 写最小 fixture。
- 或把稳定 fixture 放到 `tests/fixtures/`，并说明来源和用途。

## 6. Patch 和注入规则

优先级：

1. 构造输入并调用 public API。
2. 通过显式注入边界替换外部系统。
3. patch 模块级稳定 wrapper。
4. 最后才 patch 深层实现细节。

禁止把测试写成实现锁：

- 不断言私有函数调用顺序，除非该私有函数就是被测对象。
- 不为了测试方便在生产代码里增加无语义的旁路。
- 不 patch 同一条调用链的多个层级，否则测试无法说明真实边界。

## 7. 删测规则

拆分前先判断测试是否值得保留。以下测试应直接删除，除非能改写成更窄的 public contract：

- 大段复刻 production 常量、schema dict、manifest 字段清单的实现镜像测试。
- 已有命令/API payload contract 覆盖后，仍逐字段重复断言同一输出结构的测试。
- 一个用例串联多个无关 route / command，只为提高覆盖率，失败后无法定位具体业务边界。
- 为了测试方便而依赖实现内部顺序、临时文件布局或私有中间状态的测试。
- 与当前主线无关、没有明确历史 bug 或用户可见 contract 支撑的边缘断言。

优先保留：

- 能复现真实 bug 的回归测试。
- 锁定 public API / CLI / config / artifact contract 的测试。
- 覆盖训练、评估、推理、导出主链装配的 smoke / component。
- 对数据语义、metric、codec、target label 这类容易误判的边界测试。

## 8. 文件拆分规则

单个测试文件超过约 600 行时，应优先拆分：

- 按被测 public API 拆。
- 按 contract / component / smoke 拆。
- 按 domain builder 抽公共场景。

当前优先观察对象：

- `tests/test_infer_engine.py`
- `tests/test_data_center.py`
- `projects/eval_bench/tests/test_dashboard_rank_suite.py`
- `projects/eval_bench/tests/test_dashboard_benchmarks.py`
- `projects/eval_bench/tests/test_dashboard_import_compare.py`

当前行数最高的观察文件：

| 文件 | 行数 | 主要问题 |
| --- | ---: | --- |
| `tests/test_infer_engine.py` | 390 | infer engine 多个 backend / request 场景集中，后续按 adapter contract 拆 |
| `tests/test_data_center.py` | 380 | data source、split、mixing 组件场景集中，后续按数据入口拆 |
| `projects/eval_bench/tests/test_dashboard_rank_suite.py` | 368 | dashboard rank/suite route contract 较集中 |
| `projects/eval_bench/tests/test_dashboard_benchmarks.py` | 366 | dashboard benchmark route contract 较集中 |
| `projects/eval_bench/tests/test_dashboard_import_compare.py` | 354 | import/compare dashboard route contract 较集中 |

已完成拆分的原风险文件：

| 原文件 | 新文件 |
| --- | --- |
| `tests/test_training_modules.py` | `tests/test_training_loss.py`, `tests/test_training_optimizer.py`, `tests/test_sft_trainer.py`, `tests/test_rlhf_utils.py` |
| `tests/test_config_loader.py` | `tests/test_config_loader.py`, `tests/test_config_deepspeed.py`, `tests/test_config_catalog.py`, `tests/test_config_online_eval.py`, `tests/test_config_examples.py`, `tests/test_config_prompt_sampling.py`, `tests/test_config_validation.py`, `tests/test_config_freeze.py`, `tests/test_config_rlhf.py` |
| `projects/eval_bench/tests/test_cli.py` | `test_cli_parser_contract.py`, `test_cli_init_run.py`, `test_cli_import_predictions.py`, `test_cli_rank_listing.py`, `test_cli_detail_commands.py`, `test_cli_run_eval_compare.py`, `test_cli_run_notes.py`, `test_cli_run_admin.py`, `test_cli_ops_summary.py`, `test_cli_templates.py`, `test_cli_jobs_manifest.py`, `test_cli_services.py` |
| `projects/eval_bench/tests/test_dashboard.py` | `test_dashboard_overview.py`, `test_dashboard_rank_suite.py`, `test_dashboard_jobs.py`, `test_dashboard_benchmarks.py`, `test_dashboard_services.py`, `test_dashboard_run_samples.py`, `test_dashboard_import_compare.py` |
| `projects/eval_bench/tests/test_worker.py` | `test_worker_runtime.py`, `test_worker_jobs.py`, `test_worker_vllm.py` |
| `tests/test_online_eval.py` | `tests/test_online_eval_runner.py`, `tests/test_online_eval_aggregation.py`, `tests/test_eval_metrics.py` |
| `projects/eval_bench/tests/test_evaluator.py` | `test_evaluator_run.py`, `test_metric_engine.py`, `test_comparison.py` |
| `tests/test_pipeline_sft.py` | `tests/test_pipeline_sft.py`, `tests/test_pipeline_training_args.py`, `tests/test_training_topology.py` |
| `tests/test_model_registry.py` | `tests/test_model_registry.py`, `tests/test_model_meta.py`, `tests/test_model_processor_policy.py`, `tests/test_model_builder_validation.py`, `tests/test_model_adapter_checkpoint.py` |
| `projects/eval_bench/tests/test_job_spec.py` | `test_job_spec_manifest.py`, `test_job_spec_runtime.py`, `test_job_spec_preflight.py`, `test_job_spec_rejections.py` |
| `tests/test_pipeline_rlhf.py` | `tests/test_pipeline_rlhf.py`, `tests/test_pipeline_rlhf_smoke.py`, `tests/test_rlhf_trl_configs.py`, `tests/test_grpo_dataset.py` |
| `projects/eval_bench/tests/test_cli_templates_jobs_services.py` | `test_cli_templates.py`, `test_cli_jobs_manifest.py`, `test_cli_services.py` |
| `projects/eval_bench/tests/test_cli_run_lifecycle.py` | `test_cli_run_eval_compare.py`, `test_cli_run_notes.py`, `test_cli_run_admin.py`, `test_cli_ops_summary.py`; `delete-service` 合并到 `test_cli_services.py` |
| `projects/eval_bench/tests/test_schema.py` | `test_prediction_schema.py`, `test_artifact_layout.py`, `test_eval_manifests.py`, `test_benchmark_creation.py`, `test_benchmark_split_resolution.py`, `test_banana_benchmark_script.py`, `test_inference_schema.py` |
| `projects/eval_bench/tests/test_cli_rank_listing.py` | `test_cli_rank_board.py`, `test_cli_listing.py`, `test_cli_target_labels.py` |
| `tests/test_config_online_eval.py` | `tests/test_config_online_eval.py`, `tests/test_config_online_eval_best_metric.py`, `tests/test_config_online_eval_validation.py`, `tests/test_config_eval_datasets.py` |
| `projects/eval_bench/tests/test_dashboard_run_samples.py` | `test_dashboard_run_sample_detail.py`, `test_dashboard_composite_samples.py`, `test_dashboard_sample_image_urls.py` |

拆分不是简单移动代码。每次拆分必须同时减少至少一种重复：

- 重复 config/YAML。
- 重复 artifact/run/benchmark 目录构造。
- 重复 patch。
- 重复 fake class。

## 9. 模块最低测试责任

| 模块 | Unit / Component | Contract / Smoke |
| --- | --- | --- |
| `config` | schema / loader / normalize / catalog 展开 | 示例 YAML contract |
| `data` | source / dataset / collator / mixing / sampler | 最小 data center 装配 |
| `model` | registry / policy / finetune plan / freeze plan | smoke model 装配 |
| `template` | chat template / supervision plan | 模板 registry contract |
| `pipeline` | 装配决策、rank0 artifact、配置转换 | SFT / RLHF smoke |
| `training` | trainer helper、optimizer、scheduler、checkpoint | CPU fake trainer smoke |
| `codec` | decode、repair、partial parse | infer / eval shared codec contract |
| `metrics` | metric primitive、aggregator | online eval score smoke |
| `infer` | loader、engine adapter、pixel budget | CLI / vLLM OpenAI contract |
| `export` | inspect / validate / merge helper | export CLI contract |
| `webui` | controller、service、config service | CLI / route contract |
| `eval_bench` | semantics、store、worker helper、metric engine | CLI/API/dashboard contract |

## 10. 变更类型与必跑清单

新增配置字段：

- `tests/test_config_loader.py` 中对应 schema/normalize 覆盖。
- 一条消费该字段的 component 或 smoke。
- `docs/config_reference.md`。

新增数据源、mixing、collator：

- data unit/component。
- 如影响训练批次，再跑 SFT smoke。

新增模型族或模板：

- model/template registry。
- smoke model 或真实模型 integration，按资源条件选择。

新增算法或 pipeline：

- 算法 component。
- pipeline component。
- `pytest -q -m smoke` 中最短链。

新增 infer / codec / online eval：

- codec unit。
- infer adapter component。
- online eval component。
- 首次 eval smoke。

新增 Eval Bench 能力：

- semantics / store / worker / metric unit 或 component。
- CLI/API contract。
- 前端 UI contract 或 layout smoke，只覆盖真实用户可见行为。

## 11. GitHub CI Gate

GitHub Actions 按职责拆分为两个 focused gate，避免 eval-bench 前端或浏览器环境阻塞训练框架合并：

- `.github/workflows/framework-ci.yml`
  - 主门禁。
  - 触发范围：`src/shaft/**`、`tests/**`、训练/推理/导出薄入口、`configs/**`、依赖文件和自身 workflow。
  - 安装 `dev + train + rlhf + distributed` extras，覆盖 SFT trainer、pipeline、RLHF 配置、
    TRL adapter 与 DeepSpeed/FSDP 配置 contract。
  - 运行 `uv run python -m compileall src/shaft tests`。
  - 运行 `uv run pytest -q tests -m "not smoke and not integration and not manual"`。
- `.github/workflows/eval-bench-backend.yml`
  - Eval Bench 后端 focused 门禁。
  - 触发范围：`projects/eval_bench/eval_bench/**`、`projects/eval_bench/tests/**`、
    `scripts/eval_bench.py`、共享 codec / pixel budget、依赖文件和自身 workflow。
  - 安装 `dev + eval-bench` extras。
  - 运行
    `uv run pytest -q projects/eval_bench/tests -m "not smoke and not integration and not manual"`。

Eval Bench 前端构建、UI contract、Playwright layout smoke 不进入默认 GitHub CI。它们用于 dashboard
前端专项 review 或发布前人工验收：

```bash
cd projects/eval_bench/frontend
npm ci
npm run build
npm run test:ui-contracts
npm run test:layout
```

真实 vLLM、GPU、长推理、Docker build 和浏览器多视口布局都属于 integration/manual 验收；业务推理环境一致性用
`docker/inference/` 中的 `shaft-contract-smoke` 记录 prompt hash、pixel budget、generation、finish reason、
shared codec parser 状态和 token usage。

## 12. 当前重构计划

阶段 1：测试边界止血。

- 注册 `unit/component/contract/smoke` marker。
- 默认 `pytest -q` 排除 smoke / integration / manual。
- 新增 `tests/support`，迁移 CLI、训练 smoke、trainer helper 的共享 builder/fake。

阶段 2：核心主链测试去重。

- 已迁移 `tests/support/configs.py`、`tests/support/cli.py`、`tests/support/training.py`，并拆出 `tests/test_config_examples.py`。
- 已拆 `tests/test_config_loader.py` 为 loader/normalize、DeepSpeed、catalog、online eval policy 和示例配置 contract。
- 已迁移 `tests/support/pipeline.py`，收敛 SFT/RLHF pipeline 测试的 fake HF runtime 和 artifact builder。
- 已拆 `tests/test_training_modules.py` 为 loss、optimizer、SFT trainer、RLHF utils 四个主题文件。
- 继续拆 `tests/test_pipeline_sft.py` 和 `tests/test_pipeline_rlhf.py` 的行为域，避免 pipeline 文件继续扩张。
- 把 CLI Namespace 和 RuntimeConfig builder 迁到 support。

阶段 3：Eval Bench 大文件拆分。

- 已建立 `projects/eval_bench/tests/support`，先迁移 JSON helper、CLI payload contract、sample store 和 job payload builder。
- 已拆 `test_cli.py` 为 parser/init-run/lifecycle/import/rank/detail/templates/jobs/services 七组 contract 文件，并删除低价值 schema 镜像测试。
- 已删除 `test_dashboard.py` 中低价值的大而全 store-state 串烧用例，并按 route family 拆成 dashboard API contract 文件。
- 引入 Eval Bench store/run/benchmark scenario builder。

阶段 4：质量门收口。

- 默认 `pytest -q` 只保留稳定核心回归。
- CI / 本地推荐命令明确区分 framework、smoke、eval-bench backend、frontend manual。
- 文档、AGENTS 和开发日志保持一致。
